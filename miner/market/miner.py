import time
import argparse
import logging
import bittensor as bt


logger = logging.getLogger(__name__)

class Miner:
    def __init__(self, interactive_mode=True, wallet_name=None, hotkey_name=None, network=None, polymarket_id=None):
        self.interactive_mode = interactive_mode
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.network = network
        self.polymarket_id = polymarket_id

    @staticmethod
    def submit_metadata_to_chain(
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        metadata_dict: dict,
        network: str = "finney",  # or "test" for testnet
        max_retries: int = 10,
        retry_delay: int = 120,  # seconds between retries
    ):
        """
        Submit metadata to the Bittensor blockchain.
        
        Args:
            wallet_name: Name of your wallet (coldkey)
            hotkey_name: Name of your hotkey
            netuid: The subnet UID you're registered on
            metadata_dict: Dictionary containing your metadata (e.g., {"polymarket_id": "0x4Cd..."})
            network: Network to connect to ("finney" for mainnet, "test" for testnet)
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds to wait between retry attempts
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        # Initialize wallet and chain client
        logger.info("Initializing wallet: %s/%s", wallet_name, hotkey_name)
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        
        # Extract polymarket ID and submit only first 5 characters
        polymarket_id = metadata_dict.get('polymarket_id', '')
        
        # IMPORTANT: Only submit first 5 characters of the polymarket ID
        # This ensures we store just the beginning of the Polygon address on-chain
        metadata_to_submit = polymarket_id[:5]
        
        logger.info("Full polymarket ID: %s", polymarket_id)
        logger.info("Submitting (first 5 chars of ID): %s", metadata_to_submit)

        logger.info("Connecting to network: %s", network)
        with bt.Subtensor(network) as subtensor:
            # Verify hotkey is registered
            try:
                uid = subtensor.neurons.uid(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    netuid=netuid,
                )
                if uid is None:
                    logger.error(
                        "Hotkey %s is not registered on subnet %s",
                        wallet.hotkey.ss58_address,
                        netuid,
                    )
                    return False
                logger.info("Hotkey verified on subnet %s with UID %s", netuid, uid)
            except Exception:
                logger.exception("Failed to verify registration")
                return False

            # Submit metadata with retry loop
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                logger.info("Attempt %s/%s to commit metadata...", attempt, max_retries)

                try:
                    call = bt.calls.Commitments.set_commitment(
                        netuid=netuid,
                        info={
                            "fields": [
                                {
                                    f"Raw{len(metadata_to_submit)}": metadata_to_submit.encode(
                                        "utf-8"
                                    )
                                }
                            ]
                        },
                    )
                    result = subtensor.submit_call(
                        call,
                        wallet,
                        signer="hotkey",
                    )

                    if result.success:
                        logger.info(
                            "Successfully committed metadata to chain: %s",
                            metadata_to_submit,
                        )
                        logger.info("Hotkey: %s", wallet.hotkey.ss58_address)
                        logger.info("Subnet: %s", netuid)
                        return True

                    logger.warning("Commit failed: %s", result.message)
                except Exception:
                    logger.exception("Failed to commit metadata")

                if attempt < max_retries:
                    logger.info("Retrying in %s seconds...", retry_delay)
                    time.sleep(retry_delay)

        logger.error("Failed to commit metadata after %s attempts", max_retries)
        return False


    @staticmethod
    def retrieve_metadata_from_chain(
        hotkey_address: str,
        netuid: int,
        network: str = "finney",
    ):
        """
        Retrieve committed metadata from the chain for a specific hotkey.
        
        Args:
            hotkey_address: The ss58 address of the hotkey
            netuid: The subnet UID
            network: Network to connect to
        
        Returns:
            str: The committed data, or None if not found
        """
        logger.info("Retrieving metadata for hotkey: %s", hotkey_address)
        try:
            with bt.Subtensor(network) as subtensor:
                uid = subtensor.neurons.uid(
                    hotkey_ss58=hotkey_address,
                    netuid=netuid,
                )
                if uid is None:
                    logger.error("Hotkey not found on subnet %s", netuid)
                    return None

                commitment = subtensor.identity.commitment(
                    netuid=netuid,
                    hotkey_ss58=hotkey_address,
                )

            if commitment is None:
                return None

            logger.info("Retrieved commitment: %s", commitment.data)
            return commitment.data
        except Exception:
            logger.exception("Failed to retrieve metadata")
            return None

    def validate_polygon_address(self, address: str) -> bool:
        """
        Basic validation for Polygon address format.
        Polygon addresses are Ethereum-compatible (0x followed by 40 hex characters).
        """
        if not address.startswith('0x'):
            return False
        if len(address) != 42:  # 0x + 40 hex chars
            return False
        try:
            int(address[2:], 16)  # Check if the rest is valid hex
            return True
        except ValueError:
            return False

    def validate_wallet_exists(self, wallet_name: str, hotkey_name: str) -> bool:
        """
        Check if the wallet and hotkey exist on the local system.
        """
        try:
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            # Try to access the hotkey to verify it exists
            _ = wallet.hotkey.ss58_address
            return True
        except Exception:
            logger.exception("Wallet validation failed")
            return False

    def validate_registration(self, wallet_name: str, hotkey_name: str, netuid: int, network: str) -> bool:
        """
        Check if the hotkey is registered on the specified subnet.
        """
        try:
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            with bt.Subtensor(network) as subtensor:
                uid = subtensor.neurons.uid(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    netuid=netuid,
                )
            return uid is not None
        except Exception:
            logger.exception("Registration validation failed")
            return False

    def get_network_netuid(self, network: str) -> int:
        """
        Get the NETUID based on the network.
        finney (mainnet) = 41, test (testnet) = 172
        """
        if network.lower() == 'finney':
            return 41
        elif network.lower() == 'test':
            return 172
        else:
            raise ValueError(f"Unknown network: {network}. Use 'finney' or 'test'")

    def interactive_setup(self):
        """
        Interactive setup process for miners.
        Prompts user for wallet name, network, and polymarket ID.
        Skips prompts if CLI arguments are already provided.
        """
        # ASCII Art Banner
        ascii_banner = """

     $$$$$$\  $$\       $$\      $$\  $$$$$$\  $$\   $$\  $$$$$$\   $$$$$$\  
    $$  __$$\ $$ |      $$$\    $$$ |$$  __$$\ $$$\  $$ |$$  __$$\ $$  __$$\ 
    $$ /  $$ |$$ |      $$$$\  $$$$ |$$ /  $$ |$$$$\ $$ |$$ /  $$ |$$ /  \__|
    $$$$$$$$ |$$ |      $$\$$\$$ $$ |$$$$$$$$ |$$ $$\$$ |$$$$$$$$ |$$ |      
    $$  __$$ |$$ |      $$ \$$$  $$ |$$  __$$ |$$ \$$$$ |$$  __$$ |$$ |      
    $$ |  $$ |$$ |      $$ |\$  /$$ |$$ |  $$ |$$ |\$$$ |$$ |  $$ |$$ |  $$\ 
    $$ |  $$ |$$$$$$$$\ $$ | \_/ $$ |$$ |  $$ |$$ | \$$ |$$ |  $$ |\$$$$$$  |
    \__|  \__|\________|\__|     \__|\__|  \__|\__|  \__|\__|  \__| \______/ 
                                                                                                                                           
                               Powered by
                   ╔═╗╔═╗╔═╗╦═╗╔╦╗╔═╗╔╦╗╔═╗╔╗╔╔═╗╔═╗╦═╗
                   ╚═╗╠═╝║ ║╠╦╝ ║ ╚═╗ ║ ║╣ ║║║╚═╗║ ║╠╦╝
                   ╚═╝╩  ╚═╝╩╚═ ╩ ╚═╝ ╩ ╚═╝╝╚╝╚═╝╚═╝╩╚═

    ________________________________________________________________________

"""
        print(ascii_banner)
        print("This script will help you configure your miner for the subnet.")
        print("You'll need to provide your wallet information and Polymarket profile ID.\n")
        
        # Get wallet name (skip if provided via CLI)
        if self.wallet_name:
            wallet_name = self.wallet_name
            print(f"✅ Wallet name: {wallet_name}")
        else:
            while True:
                wallet_name = input("Enter your Bittensor wallet name (coldkey): ").strip()
                if wallet_name:
                    break
                print("❌ Wallet name cannot be empty. Please try again.")
        
        # Get hotkey name (skip if provided via CLI)
        if self.hotkey_name:
            hotkey_name = self.hotkey_name
            print(f"✅ Hotkey name: {hotkey_name}")
        else:
            hotkey_name = input("Enter your hotkey name (press Enter to use same as wallet name): ").strip()
            if not hotkey_name:
                hotkey_name = wallet_name
            print(f"✅ Hotkey name: {hotkey_name}")
        
        # Get network (skip if provided via CLI)
        if self.network:
            network = self.network.lower()
            if network == 'finney':
                netuid = 41
            elif network == 'test':
                netuid = 172
            else:
                print(f"❌ Invalid network from CLI: {network}. Use 'finney' or 'test'")
                return False
            print(f"✅ Network: {network} (NETUID: {netuid})")
        else:
            while True:
                print("\nSelect network:")
                print("1. finney (mainnet) - NETUID 41")
                print("2. test (testnet) - NETUID 172")
                network_choice = input("Enter choice (1 or 2): ").strip()
                
                if network_choice == '1':
                    network = 'finney'
                    netuid = 41
                    break
                elif network_choice == '2':
                    network = 'test'
                    netuid = 172
                    break
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
            print(f"\n✅ Selected network: {network} (NETUID: {netuid})")
        
        # Now validate wallet and registration before asking for polymarket ID
        print(f"\n🔍 Validating wallet and registration...")
        
        # Validate wallet exists
        print("🔍 Checking wallet...")
        if not self.validate_wallet_exists(wallet_name, hotkey_name):
            print(f"❌ Wallet '{wallet_name}' with hotkey '{hotkey_name}' not found. Please make sure your wallet is properly set up.")
            print("You can create a wallet using: btcli wallet new-coldkey --wallet <name>")
            print("You can create a hotkey using: btcli wallet new-hotkey --wallet <name> --wallet-hotkey <hotkey_name>")
            return False
        
        print("✅ Wallet found!")
        
        # Validate registration
        print(f"🔍 Checking registration on subnet {netuid}...")
        if not self.validate_registration(wallet_name, hotkey_name, netuid, network):
            print(f"❌ Hotkey not registered on subnet {netuid} ({network} network).")
            print(f"Please register using: btcli subnets register --netuid {netuid} --network {network} --wallet {wallet_name} --wallet-hotkey {hotkey_name}")
            return False
        
        print("✅ Hotkey is registered on the subnet!")
        
        # Check for existing metadata
        print(f"🔍 Checking for existing metadata...")
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        existing_commitment = self.retrieve_metadata_from_chain(
            hotkey_address=wallet.hotkey.ss58_address,
            netuid=netuid,
            network=network,
        )
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"⚠️  Existing metadata found: {existing_commitment}")
            print("Note: Only the first 5 characters of your Almanac/Polymarket Ethereum address (EOA) are stored on-chain for privacy.")
        else:
            print("✅ No existing metadata found.")
        
        # Now ask if user wants to proceed with metadata submission
        print("\n" + "="*60)
        print("📋 VALIDATION COMPLETE")
        print("="*60)
        print(f"Wallet Name: {wallet_name}")
        print(f"Hotkey Name: {hotkey_name}")
        print(f"Network: {network}")
        print(f"NETUID: {netuid}")
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"Existing Metadata: {existing_commitment}")
        print("="*60)
        
        proceed = input("\nDo you want to submit/update your Almanac/Polymarket Ethereum address (EOA) metadata? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("✅ Setup completed. No metadata changes made.")
            return True  # Return True since validation was successful
        
        # Get polymarket ID (skip if provided via CLI)
        if self.polymarket_id:
            polymarket_id = self.polymarket_id
            print(f"\n✅ Polymarket ID: {polymarket_id}")
        else:
            while True:
                polymarket_id = input("\nEnter your Almanac/Polymarket Ethereum address (EOA) starting with 0x: ").strip()
                if self.validate_polygon_address(polymarket_id):
                    break
                print("❌ Invalid EOA address format. Please enter a valid address (0x followed by 40 hex characters).")
            print(f"\n✅ EOA Address: {polymarket_id}")
        
        # Final confirmation for metadata submission
        print("\n" + "="*60)
        print("📋 METADATA SUBMISSION SUMMARY")
        print("="*60)
        print(f"Wallet Name: {wallet_name}")
        print(f"Hotkey Name: {hotkey_name}")
        print(f"Network: {network}")
        print(f"NETUID: {netuid}")
        print(f"Polymarket ID: {polymarket_id}")
        print("="*60)
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            confirm = input("\nProceed with metadata update? (y/N): ").strip().lower()
        else:
            confirm = input("\nProceed with metadata submission? (y/N): ").strip().lower()
            
        if confirm != 'y' and confirm != 'yes':
            print("❌ Metadata submission cancelled.")
            return True  # Return True since validation was successful
        
        # Submit metadata
        print("\n🚀 Submitting metadata to blockchain...")
        metadata = {
            "polymarket_id": polymarket_id
        }
        
        success = self.submit_metadata_to_chain(
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            netuid=netuid,
            metadata_dict=metadata,
            network=network,
        )
        
        if success:
            print("\n✅ Metadata successfully committed to chain!")
            
            # Retrieve and display the commitment
            retrieved = self.retrieve_metadata_from_chain(
                hotkey_address=wallet.hotkey.ss58_address,
                netuid=netuid,
                network=network,
            )
            print(f"Retrieved commitment: {retrieved}")
            print("\n📝 Note: Only the first 5 characters of your Almanac/Polymarket ID are stored on-chain for privacy.")
            print(f"Your full Almanac/Polymarket ID: {polymarket_id}")
            print(f"Stored on-chain: {polymarket_id[:5]}")
            print("\n🎉 Setup complete! Your miner is now configured.")
            return True
        else:
            print("\n❌ Failed to commit metadata. Please try again later.")
            return False

    def run_miner_setup(self):
        """
        Legacy setup function - now calls interactive setup.
        """
        return self.interactive_setup()

def main():
    """
    Main function that handles both interactive and CLI modes.
    """
    parser = argparse.ArgumentParser(description='SN41 Almanac Miner Setup')
    parser.add_argument(
        "--wallet", "-w", "--wallet.name", dest="wallet_name", help="Wallet name (coldkey)"
    )
    parser.add_argument(
        "--wallet-hotkey", "-H", "--wallet.hotkey", dest="hotkey_name", help="Hotkey name"
    )
    parser.add_argument(
        "--network", "-n", "--subtensor.network", dest="network", help="Network (finney or test)"
    )
    parser.add_argument('--polymarket.id', dest='polymarket_id', help='Almanac/Polymarket ID (EOA address)')
    
    args = parser.parse_args()
    
    # Check if we have all required CLI arguments for full CLI mode
    has_all_cli_args = args.wallet_name and args.hotkey_name and args.polymarket_id and args.network
    
    if has_all_cli_args:
        # Full CLI mode - no interactive prompts
        print("🔧 Running in CLI mode...")
        
        # Validate network and get NETUID
        if args.network.lower() == 'finney':
            netuid = 41
        elif args.network.lower() == 'test':
            netuid = 172
        else:
            print(f"❌ Invalid network: {args.network}. Use 'finney' or 'test'")
            exit(1)
        
        # Create miner instance for CLI mode
        miner = Miner(interactive_mode=False, wallet_name=args.wallet_name, hotkey_name=args.hotkey_name)
        
        # Validate inputs
        if not miner.validate_polygon_address(args.polymarket_id):
            print("❌ Invalid Polygon address format.")
            exit(1)
        
        # Validate wallet exists
        if not miner.validate_wallet_exists(args.wallet_name, args.hotkey_name):
            print(f"❌ Wallet '{args.wallet_name}' with hotkey '{args.hotkey_name}' not found.")
            exit(1)
        
        # Validate registration
        if not miner.validate_registration(args.wallet_name, args.hotkey_name, netuid, args.network):
            print(f"❌ Hotkey not registered on subnet {netuid} ({args.network} network).")
            exit(1)
        
        # Check for existing metadata in CLI mode
        print("🔍 Checking for existing metadata...")
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey_name)
        existing_commitment = miner.retrieve_metadata_from_chain(
            hotkey_address=wallet.hotkey.ss58_address,
            netuid=netuid,
            network=args.network,
        )
        
        if existing_commitment and existing_commitment != 'None' and existing_commitment.strip():
            print(f"⚠️  Existing metadata found: {existing_commitment}")
            print("Note: Only the first 5 characters of the original Polymarket ID are stored on-chain.")
            print("Use interactive mode if you want to choose whether to overwrite existing metadata.")
        else:
            print("✅ No existing metadata found.")
        
        # Submit metadata
        print("\n🚀 Submitting metadata to blockchain...")
        metadata = {"polymarket_id": args.polymarket_id}
        
        success = miner.submit_metadata_to_chain(
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            netuid=netuid,
            metadata_dict=metadata,
            network=args.network,
        )
        
        if success:
            print("✅ Metadata successfully committed to chain!")
            retrieved = miner.retrieve_metadata_from_chain(
                hotkey_address=wallet.hotkey.ss58_address,
                netuid=netuid,
                network=args.network,
            )
            print(f"Retrieved commitment: {retrieved}")
            print("\n📝 Note: Only the first 5 characters of your Polymarket ID are stored on-chain due to blockchain constraints.")
            print(f"Your full Polymarket ID: {args.polymarket_id}")
            print(f"Stored on-chain: {args.polymarket_id[:5]}")
            print("🎉 Setup complete!")
        else:
            print("❌ Failed to commit metadata.")
            exit(1)
    else:
        # Interactive mode with optional CLI arguments
        try:
            miner = Miner(
                interactive_mode=True, 
                wallet_name=args.wallet_name, 
                hotkey_name=args.hotkey_name,
                network=args.network,
                polymarket_id=args.polymarket_id
            )
            success = miner.run_miner_setup()
            if not success:
                exit(1)
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user.")
            exit(1)
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            logger.exception("Setup failed")
            exit(1)

# Run the miner.
if __name__ == "__main__":
    main()
