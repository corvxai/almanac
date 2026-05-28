import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import bittensor as bt


class MetadataManager:
    """
    Manages metadata retrieval and local storage for subnet participants.
    Runs in a background thread to avoid blocking the main validator loop.
    """
    
    def __init__(
        self,
        netuid: int,
        network: str,
        state_file: str = "validator_state.json",
        update_interval_seconds: int = 3600,
        batch_size: int = 10,
        batch_delay_seconds: float = 2.0,
        per_uid_delay_seconds: float = 0.1,
        use_bulk_commitments: bool = True,
    ):
        self.netuid = netuid
        self.network = network
        self.state_file = state_file
        self.subtensor = bt.Subtensor(network=network)
        self.metagraph = None
        
        # Configuration
        self.update_interval = max(1, int(update_interval_seconds))
        self.batch_size = max(1, int(batch_size))
        self.batch_delay = max(0.0, float(batch_delay_seconds))
        self.per_uid_delay = max(0.0, float(per_uid_delay_seconds))
        self.use_bulk_commitments = bool(use_bulk_commitments)
        
        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Load existing state
        self.metadata_state = self.load_state()
        
        bt.logging.info(f"MetadataManager initialized for subnet {netuid} on {network}")
    
    def load_state(self) -> Dict:
        """Load metadata state from JSON file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                bt.logging.info(f"Loaded metadata state with {len(state.get('metadata', []))} entries")
                return state
            except Exception as e:
                bt.logging.warning(f"Failed to load state file {self.state_file}: {e}")
        
        return {
            "metadata": [],
            "last_full_sync": None,
            "version": "1.0"
        }
    
    def save_state(self):
        """Save metadata state to JSON file."""
        try:
            with self.lock:
                with open(self.state_file, 'w') as f:
                    json.dump(self.metadata_state, f, indent=2)
        except Exception as e:
            bt.logging.error(f"Failed to save state file {self.state_file}: {e}")
    
    def sync_metagraph(self):
        """Sync metagraph to get current UIDs."""
        try:
            self.metagraph = self.subtensor.metagraph(self.netuid)
            bt.logging.debug(f"Synced metagraph: {len(self.metagraph.uids)} total UIDs")
        except Exception as e:
            bt.logging.error(f"Failed to sync metagraph: {e}")
    
    def get_uid_metadata(self, uid: int) -> Optional[str]:
        """Get metadata for a specific UID from blockchain."""
        import logging
        
        # Temporarily suppress bittensor logging to avoid NoneType errors
        bt_logger = logging.getLogger('bittensor')
        subtensor_logger = logging.getLogger('bittensor.subtensor')
        
        original_bt_level = bt_logger.getEffectiveLevel()
        original_subtensor_level = subtensor_logger.getEffectiveLevel()
        
        bt_logger.setLevel(logging.CRITICAL)  # Only show critical errors
        subtensor_logger.setLevel(logging.CRITICAL)  # Suppress subtensor errors
        
        try:
            commitment = self.subtensor.get_commitment(netuid=self.netuid, uid=uid)
            if commitment and commitment.strip():
                return commitment
            # Return None explicitly if no metadata (this is normal)
            return None
            
        except Exception as e:
            bt.logging.debug(f"Failed to get metadata for UID {uid}: {e}")
            return None
        finally:
            # Restore original logging levels
            bt_logger.setLevel(original_bt_level)
            subtensor_logger.setLevel(original_subtensor_level)
    
    def get_uid_info(self, uid: int) -> Optional[Dict]:
        """Get existing metadata info for a UID from local state."""
        for entry in self.metadata_state["metadata"]:
            if entry["uid"] == uid:
                return entry
        return None
    
    def update_uid_metadata(self, uid: int, polymarket_id: Optional[str], block_number: int):
        """Update or add metadata for a UID."""
        with self.lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            # Find existing entry
            for i, entry in enumerate(self.metadata_state["metadata"]):
                if entry["uid"] == uid:
                    # Update existing entry
                    self.metadata_state["metadata"][i].update({
                        "polymarket_id": polymarket_id.lower() if polymarket_id is not None else None,
                        "last_updated": timestamp
                    })
                    return
            
            # Add new entry (polymarket_id can be None if no metadata found)
            self.metadata_state["metadata"].append({
                "uid": uid,
                "polymarket_id": polymarket_id.lower() if polymarket_id is not None else None,
                "last_updated": timestamp
            })
    
    def get_uids_to_update(self) -> List[int]:
        """Get list of UIDs that need metadata updates."""
        if not self.metagraph:
            return []
        
        current_time = datetime.now(timezone.utc)
        uids_to_update = []
        
        # Get all non-validator UIDs
        all_uids = set(self.metagraph.uids.tolist())
        
        # Get validator UIDs by checking which UIDs have validator permits
        """
        validator_uids = set()
        try:
            # Query validator permits for all UIDs in the subnet
            validator_permit_data = self.subtensor.query_map(
                'SubtensorModule', 'ValidatorPermit', [self.netuid]
            )
            
            for item in validator_permit_data:
                # Handle both tuple and list formats from query_map
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    key, permit = item[0], item[1]
                    if isinstance(key, (tuple, list)) and len(key) >= 2:
                        netuid_key, uid = key[0], key[1]
                        if permit:  # If permit is True
                            validator_uids.add(uid)
                    
        except Exception as e:
            bt.logging.warning(f"Could not query validator permits: {e}")
            # Fallback: assume no validators if we can't query
            validator_uids = set()
        """
        
        #miner_uids = all_uids - validator_uids
        
        for uid in all_uids:
            uid_info = self.get_uid_info(uid)
            
            # Update if:
            # 1. No local data exists
            # 2. Last update was more than update_interval ago
            should_update = False
            
            if not uid_info:
                should_update = True
            else:
                try:
                    last_updated_str = uid_info.get("last_updated")
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        # Make timezone-aware if naive (assume UTC for backward compatibility)
                        if last_updated.tzinfo is None:
                            last_updated = last_updated.replace(tzinfo=timezone.utc)
                        time_since_update = (current_time - last_updated).total_seconds()
                        
                        if time_since_update > self.update_interval:
                            should_update = True
                    else:
                        should_update = True
                except (ValueError, TypeError):
                    # Invalid timestamp format, update anyway
                    should_update = True
            
            if should_update:
                uids_to_update.append(uid)
        
        bt.logging.debug(f"Found {len(uids_to_update)} UIDs to update out of {len(all_uids)} total miners")
        return uids_to_update
    
    def _normalize_commitment(self, commitment: Optional[str]) -> Optional[str]:
        if commitment is None:
            return None
        if isinstance(commitment, str):
            commitment = commitment.strip()
            return commitment or None
        return None

    def _get_bulk_commitments(self) -> Optional[Dict[int, Optional[str]]]:
        """Fetch all subnet commitments and map by UID in one chain call."""
        if self.metagraph is None:
            return None
        try:
            commitments_by_hotkey = self.subtensor.get_all_commitments(netuid=self.netuid)
            if not isinstance(commitments_by_hotkey, dict):
                bt.logging.warning("Bulk commitment fetch returned unexpected payload; falling back to per-UID fetch.")
                return None

            uid_commitments: Dict[int, Optional[str]] = {}
            hotkeys = list(self.metagraph.hotkeys)
            for uid in self.metagraph.uids.tolist():
                uid_int = int(uid)
                if uid_int < 0 or uid_int >= len(hotkeys):
                    uid_commitments[uid_int] = None
                    continue
                commitment = commitments_by_hotkey.get(hotkeys[uid_int])
                uid_commitments[uid_int] = self._normalize_commitment(commitment)

            bt.logging.debug(f"Fetched {len(commitments_by_hotkey)} commitments with bulk chain query.")
            return uid_commitments
        except Exception as e:
            bt.logging.warning(f"Bulk commitment fetch failed; falling back to per-UID fetch: {e}")
            return None

    def process_batch(
        self,
        uid_batch: List[int],
        current_block: int,
        bulk_commitments: Optional[Dict[int, Optional[str]]] = None,
    ):
        """Process a batch of UIDs for metadata updates."""
        for uid in uid_batch:
            try:
                metadata = (
                    bulk_commitments.get(uid)
                    if bulk_commitments is not None
                    else self.get_uid_metadata(uid)
                )
                # Always update the entry, even if metadata is None
                self.update_uid_metadata(uid, metadata, current_block)
                
                if metadata:
                    bt.logging.debug(f"Metadata found! Updated metadata for UID {uid}: [REDACTED]")
                else:
                    bt.logging.debug(f"No metadata found for UID {uid} (stored as None)")
                
                if bulk_commitments is None and self.per_uid_delay > 0:
                    # Small delay only for per-UID chain queries to avoid burst load.
                    time.sleep(self.per_uid_delay)
                
            except Exception as e:
                bt.logging.warning(f"Error processing UID {uid}: {e}")
                # Store None for failed queries too
                self.update_uid_metadata(uid, None, current_block)
    
    def update_metadata_batch(self):
        """Update metadata for a batch of UIDs."""
        try:
            self.sync_metagraph()
            if not self.metagraph:
                return
            
            current_block = self.metagraph.block.item()
            uids_to_update = self.get_uids_to_update()
            
            if not uids_to_update:
                bt.logging.debug("No UIDs need metadata updates")
                return

            bulk_commitments = self._get_bulk_commitments() if self.use_bulk_commitments else None
            
            # Process in batches
            for i in range(0, len(uids_to_update), self.batch_size):
                batch = uids_to_update[i:i + self.batch_size]
                bt.logging.info(f"Processing metadata batch {i//self.batch_size + 1}: UIDs {batch}")
                
                self.process_batch(batch, current_block, bulk_commitments=bulk_commitments)
                
                # Save state after each batch
                self.save_state()
                
                # Delay between batches
                if i + self.batch_size < len(uids_to_update):
                    time.sleep(self.batch_delay)
            
            # Update last full sync timestamp
            self.metadata_state["last_full_sync"] = datetime.now(timezone.utc).isoformat()
            self.save_state()
            
            bt.logging.info(f"Completed metadata update for {len(uids_to_update)} UIDs")
            
        except Exception as e:
            bt.logging.error(f"Error in metadata batch update: {e}")
    
    def background_update_loop(self):
        """Background thread loop for periodic metadata updates."""
        bt.logging.info("Starting metadata background update loop")
        
        while self.running:
            try:
                self.update_metadata_batch()
                
                # Sleep until next update cycle
                time.sleep(self.update_interval)
                
            except Exception as e:
                bt.logging.error(f"Error in background update loop: {e}")
                time.sleep(60)  # Short sleep on error
    
    def start(self):
        """Start the background metadata update thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.background_update_loop, daemon=True)
        self.thread.start()
        bt.logging.info("Metadata background thread started")
    
    def stop(self):
        """Stop the background metadata update thread."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        bt.logging.info("Metadata background thread stopped")
    
    def get_miner_metadata(self, uid: int) -> Optional[str]:
        """Get metadata for a specific miner UID (thread-safe)."""
        with self.lock:
            uid_info = self.get_uid_info(uid)
            if uid_info:
                if uid_info.get("polymarket_id") is None:
                    return None
                return uid_info.get("polymarket_id").lower()
        return None
    
    def get_all_miner_metadata(self) -> Dict[int, str]:
        """Get all miner metadata (thread-safe)."""
        with self.lock:
            metadata_dict = {}
            for entry in self.metadata_state["metadata"]:
                if entry.get("polymarket_id") is None:
                    continue
                metadata_dict[entry["uid"]] = entry["polymarket_id"].lower()
            return metadata_dict
    
    def get_stats(self) -> Dict:
        """Get metadata manager statistics."""
        with self.lock:
            total_entries = len(self.metadata_state["metadata"])
            last_sync = self.metadata_state.get("last_full_sync")
            
            return {
                "total_metadata_entries": total_entries,
                "last_full_sync": last_sync,
                "update_interval_seconds": self.update_interval,
                "batch_size": self.batch_size,
                "batch_delay_seconds": self.batch_delay,
                "per_uid_delay_seconds": self.per_uid_delay,
                "use_bulk_commitments": self.use_bulk_commitments,
                "running": self.running
            }
