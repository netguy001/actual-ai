"""
Online updater module for fetching updates and new datasets.
Handles synchronization between local and remote data/models.
"""

import os
import json
import requests
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineUpdater:
    """
    Manages online updates, dataset synchronization, and remote model updates.
    Provides secure and efficient updating capabilities for the local AI system.
    """

    def __init__(self, storage_path: str = "storage", config: Optional[Dict] = None):
        self.storage_path = storage_path
        self.config = config or self._load_default_config()
        self.session = requests.Session()
        self.last_update_check = None
        self.update_log_path = os.path.join(storage_path, "update_log.json")

        # Setup session headers
        self.session.headers.update(
            {
                "User-Agent": f'LocalAI-System/{self.config.get("version", "1.0.0")}',
                "Content-Type": "application/json",
            }
        )

        # Initialize update tracking
        self._initialize_update_tracking()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for online updates."""
        return {
            "base_url": "https://api.example.com",  # Replace with actual API endpoint
            "version": "1.0.0",
            "update_interval_hours": 24,
            "timeout": 30,
            "max_retries": 3,
            "chunk_size": 8192,
            "verify_ssl": True,
            "api_key": None,  # Set this in production
            "endpoints": {
                "version_check": "/version",
                "datasets": "/datasets",
                "models": "/models",
                "updates": "/updates",
            },
        }

    def _initialize_update_tracking(self):
        """Initialize update tracking system."""
        if not os.path.exists(self.update_log_path):
            initial_log = {
                "last_check": None,
                "last_successful_update": None,
                "update_history": [],
                "failed_updates": [],
                "local_version": self.config["version"],
            }
            with open(self.update_log_path, "w") as f:
                json.dump(initial_log, f, indent=2)

    def _get_update_log(self) -> Dict[str, Any]:
        """Load update log from disk."""
        with open(self.update_log_path, "r") as f:
            return json.load(f)

    def _save_update_log(self, log_data: Dict[str, Any]):
        """Save update log to disk."""
        with open(self.update_log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Make HTTP request with error handling and retries.

        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data for POST/PUT
            params: URL parameters

        Returns:
            Response data or None if failed
        """
        url = urljoin(self.config["base_url"], endpoint)
        headers = {}

        # Add API key if configured
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"

        for attempt in range(self.config["max_retries"]):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.config["timeout"],
                        verify=self.config["verify_ssl"],
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url,
                        json=data,
                        headers=headers,
                        timeout=self.config["timeout"],
                        verify=self.config["verify_ssl"],
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All {self.config['max_retries']} attempts failed for {url}"
                    )
                    return None

        return None

    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for available updates from the remote server.

        Returns:
            Update information including available updates
        """
        logger.info("Checking for updates...")

        current_time = datetime.now()
        log_data = self._get_update_log()

        # Check if enough time has passed since last check
        if log_data.get("last_check"):
            last_check = datetime.fromisoformat(log_data["last_check"])
            time_diff = current_time - last_check
            if time_diff < timedelta(hours=self.config["update_interval_hours"]):
                logger.info("Update check skipped - too soon since last check")
                return {"status": "skipped", "reason": "too_soon"}

        # Make version check request
        version_info = self._make_request(self.config["endpoints"]["version_check"])

        if not version_info:
            log_data["failed_updates"].append(
                {
                    "timestamp": current_time.isoformat(),
                    "error": "Failed to connect to update server",
                    "type": "version_check",
                }
            )
            self._save_update_log(log_data)
            return {"status": "error", "message": "Failed to connect to update server"}

        # Update log
        log_data["last_check"] = current_time.isoformat()
        self._save_update_log(log_data)

        # Compare versions
        remote_version = version_info.get("version", "0.0.0")
        local_version = log_data.get("local_version", "0.0.0")

        updates_available = {
            "models": version_info.get("models_updated", False),
            "datasets": version_info.get("datasets_updated", False),
            "system": self._compare_versions(local_version, remote_version) < 0,
        }

        logger.info(f"Update check completed. Available updates: {updates_available}")

        return {
            "status": "success",
            "local_version": local_version,
            "remote_version": remote_version,
            "updates_available": updates_available,
            "last_check": current_time.isoformat(),
        }

    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """

        def version_tuple(v):
            return tuple(map(int, v.split(".")))

        v1, v2 = version_tuple(version1), version_tuple(version2)
        return -1 if v1 < v2 else (1 if v1 > v2 else 0)

    def fetch_dataset_updates(self) -> Dict[str, Any]:
        """
        Fetch new datasets or dataset updates from the remote server.

        Returns:
            Update results including downloaded datasets
        """
        logger.info("Fetching dataset updates...")

        # Get available datasets
        datasets_info = self._make_request(self.config["endpoints"]["datasets"])

        if not datasets_info:
            return {"status": "error", "message": "Failed to fetch dataset information"}

        available_datasets = datasets_info.get("datasets", [])
        downloaded_datasets = []

        for dataset in available_datasets:
            dataset_name = dataset.get("name")
            dataset_url = dataset.get("download_url")
            dataset_hash = dataset.get("hash")

            if not all([dataset_name, dataset_url, dataset_hash]):
                logger.warning(f"Incomplete dataset information: {dataset}")
                continue

            # Check if dataset already exists locally
            local_path = os.path.join(
                self.storage_path, "datasets", f"{dataset_name}.json"
            )

            if os.path.exists(local_path):
                # Verify hash to check if update needed
                if self._verify_file_hash(local_path, dataset_hash):
                    logger.info(f"Dataset {dataset_name} is up to date")
                    continue

            # Download dataset
            if self._download_file(dataset_url, local_path, dataset_hash):
                downloaded_datasets.append(dataset_name)
                logger.info(f"Downloaded dataset: {dataset_name}")
            else:
                logger.error(f"Failed to download dataset: {dataset_name}")

        return {
            "status": "success",
            "downloaded_datasets": downloaded_datasets,
            "total_available": len(available_datasets),
            "timestamp": datetime.now().isoformat(),
        }

    def fetch_model_updates(self) -> Dict[str, Any]:
        """
        Fetch model updates from the remote server.

        Returns:
            Update results including downloaded models
        """
        logger.info("Fetching model updates...")

        # Get available models
        models_info = self._make_request(self.config["endpoints"]["models"])

        if not models_info:
            return {"status": "error", "message": "Failed to fetch model information"}

        available_models = models_info.get("models", [])
        downloaded_models = []

        for model in available_models:
            model_name = model.get("name")
            model_url = model.get("download_url")
            model_hash = model.get("hash")
            model_version = model.get("version")

            if not all([model_name, model_url, model_hash]):
                logger.warning(f"Incomplete model information: {model}")
                continue

            # Download model
            models_dir = os.path.join(self.storage_path, "models", "remote")
            os.makedirs(models_dir, exist_ok=True)

            local_path = os.path.join(models_dir, f"{model_name}_v{model_version}.pkl")

            if self._download_file(model_url, local_path, model_hash):
                downloaded_models.append(
                    {"name": model_name, "version": model_version, "path": local_path}
                )
                logger.info(f"Downloaded model: {model_name} v{model_version}")
            else:
                logger.error(f"Failed to download model: {model_name}")

        return {
            "status": "success",
            "downloaded_models": downloaded_models,
            "total_available": len(available_models),
            "timestamp": datetime.now().isoformat(),
        }

    def _download_file(self, url: str, local_path: str, expected_hash: str) -> bool:
        """
        Download file with hash verification.

        Args:
            url: File URL
            local_path: Local file path
            expected_hash: Expected SHA256 hash

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            response = self.session.get(
                url, stream=True, verify=self.config["verify_ssl"]
            )
            response.raise_for_status()

            hash_sha256 = hashlib.sha256()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=self.config["chunk_size"]
                ):
                    if chunk:
                        f.write(chunk)
                        hash_sha256.update(chunk)

            # Verify hash
            calculated_hash = hash_sha256.hexdigest()
            if calculated_hash != expected_hash:
                logger.error(
                    f"Hash mismatch for {local_path}. Expected: {expected_hash}, Got: {calculated_hash}"
                )
                os.remove(local_path)  # Remove corrupted file
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False

    def _verify_file_hash(self, file_path: str, expected_hash: str) -> bool:
        """Verify file hash matches expected value."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(self.config["chunk_size"]), b""):
                    hash_sha256.update(chunk)

            calculated_hash = hash_sha256.hexdigest()
            return calculated_hash == expected_hash

        except Exception as e:
            logger.error(f"Failed to verify hash for {file_path}: {e}")
            return False

    def sync_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync user learning data with remote server (optional, privacy-aware).

        Args:
            user_data: Anonymized user learning data

        Returns:
            Sync results
        """
        logger.info("Syncing user data...")

        # Prepare anonymized data for sync
        sync_data = {
            "timestamp": datetime.now().isoformat(),
            "data_hash": hashlib.sha256(
                json.dumps(user_data, sort_keys=True).encode()
            ).hexdigest(),
            "local_version": self.config["version"],
            "learning_stats": {
                "total_interactions": user_data.get("total_interactions", 0),
                "model_updates": user_data.get("model_updates", 0),
                "data_types": user_data.get("data_types", []),
            },
        }

        response = self._make_request(
            self.config["endpoints"]["updates"], method="POST", data=sync_data
        )

        if response:
            logger.info("User data sync completed successfully")
            return {"status": "success", "sync_id": response.get("sync_id")}
        else:
            logger.error("Failed to sync user data")
            return {"status": "error", "message": "Failed to sync with remote server"}

    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status and history."""
        log_data = self._get_update_log()

        return {
            "local_version": log_data.get("local_version"),
            "last_check": log_data.get("last_check"),
            "last_successful_update": log_data.get("last_successful_update"),
            "update_history_count": len(log_data.get("update_history", [])),
            "failed_updates_count": len(log_data.get("failed_updates", [])),
            "next_scheduled_check": self._calculate_next_check_time(),
        }

    def _calculate_next_check_time(self) -> Optional[str]:
        """Calculate next scheduled update check time."""
        log_data = self._get_update_log()
        last_check = log_data.get("last_check")

        if last_check:
            last_check_time = datetime.fromisoformat(last_check)
            next_check = last_check_time + timedelta(
                hours=self.config["update_interval_hours"]
            )
            return next_check.isoformat()

        return None

    def perform_full_update(self) -> Dict[str, Any]:
        """
        Perform a complete system update including datasets and models.

        Returns:
            Complete update results
        """
        logger.info("Starting full system update...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "version_check": {},
            "dataset_updates": {},
            "model_updates": {},
            "overall_status": "success",
        }

        # Check for updates
        version_result = self.check_for_updates()
        results["version_check"] = version_result

        if version_result.get("status") != "success":
            results["overall_status"] = "failed"
            return results

        updates_available = version_result.get("updates_available", {})

        # Update datasets if needed
        if updates_available.get("datasets"):
            dataset_result = self.fetch_dataset_updates()
            results["dataset_updates"] = dataset_result
            if dataset_result.get("status") != "success":
                results["overall_status"] = "partial_failure"

        # Update models if needed
        if updates_available.get("models"):
            model_result = self.fetch_model_updates()
            results["model_updates"] = model_result
            if model_result.get("status") != "success":
                results["overall_status"] = "partial_failure"

        # Log update completion
        log_data = self._get_update_log()
        log_data["last_successful_update"] = datetime.now().isoformat()
        log_data["update_history"].append(results)

        # Keep only last 50 update records
        if len(log_data["update_history"]) > 50:
            log_data["update_history"] = log_data["update_history"][-50:]

        self._save_update_log(log_data)

        logger.info(f"Full update completed with status: {results['overall_status']}")
        return results

    def rollback_update(self, update_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback to previous version (placeholder for future implementation).

        Args:
            update_id: Specific update to rollback (optional)

        Returns:
            Rollback results
        """
        logger.warning("Rollback functionality not yet implemented")
        return {
            "status": "not_implemented",
            "message": "Rollback functionality is planned for future versions",
            "timestamp": datetime.now().isoformat(),
        }

    def configure_update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration settings for the updater.

        Args:
            settings: New configuration settings

        Returns:
            Configuration update results
        """
        logger.info("Updating configuration settings...")

        allowed_settings = [
            "update_interval_hours",
            "timeout",
            "max_retries",
            "verify_ssl",
            "base_url",
            "api_key",
        ]

        updated_settings = []
        for key, value in settings.items():
            if key in allowed_settings:
                old_value = self.config.get(key)
                self.config[key] = value
                updated_settings.append(
                    {"setting": key, "old_value": old_value, "new_value": value}
                )
                logger.info(f"Updated {key}: {old_value} -> {value}")

        # Save configuration (in production, this should persist to a config file)
        config_path = os.path.join(self.storage_path, "updater_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        return {
            "status": "success",
            "updated_settings": updated_settings,
            "timestamp": datetime.now().isoformat(),
        }

    def get_network_status(self) -> Dict[str, Any]:
        """
        Check network connectivity and server availability.

        Returns:
            Network status information
        """
        logger.info("Checking network status...")

        status = {
            "timestamp": datetime.now().isoformat(),
            "server_reachable": False,
            "response_time_ms": None,
            "ssl_valid": None,
            "api_accessible": False,
        }

        try:
            start_time = time.time()
            response = self.session.get(
                urljoin(self.config["base_url"], "/health"),
                timeout=5,
                verify=self.config["verify_ssl"],
            )
            end_time = time.time()

            status["server_reachable"] = True
            status["response_time_ms"] = int((end_time - start_time) * 1000)
            status["ssl_valid"] = True
            status["api_accessible"] = response.status_code == 200

        except requests.exceptions.SSLError:
            status["ssl_valid"] = False
        except requests.exceptions.ConnectionError:
            status["server_reachable"] = False
        except requests.exceptions.Timeout:
            status["server_reachable"] = True
            status["response_time_ms"] = "timeout"
        except Exception as e:
            logger.error(f"Network check failed: {e}")

        return status

    def cleanup_old_files(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old update files and logs.

        Args:
            days_to_keep: Number of days to keep files

        Returns:
            Cleanup results
        """
        logger.info(f"Cleaning up files older than {days_to_keep} days...")

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = []

        # Clean up old datasets
        datasets_dir = os.path.join(self.storage_path, "datasets")
        if os.path.exists(datasets_dir):
            for filename in os.listdir(datasets_dir):
                file_path = os.path.join(datasets_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        cleaned_files.append(file_path)

        # Clean up old remote models
        remote_models_dir = os.path.join(self.storage_path, "models", "remote")
        if os.path.exists(remote_models_dir):
            for filename in os.listdir(remote_models_dir):
                file_path = os.path.join(remote_models_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        cleaned_files.append(file_path)

        return {
            "status": "success",
            "cleaned_files": len(cleaned_files),
            "files_list": cleaned_files,
            "cutoff_date": cutoff_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
