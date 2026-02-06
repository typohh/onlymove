import os
import subprocess
import sys

def fetch_database():
    """
    Downloads the latest database from Cloudflare D1 using wrangler.
    The command run is: npx wrangler d1 export DB --remote --output=../attempts/database.sql
    It must be run from the 'pwa' directory context.
    """
    # Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pwa_dir = os.path.join(project_root, "pwa")
    attempts_dir = os.path.join(project_root, "attempts")
    output_file = os.path.join(attempts_dir, "database.sql")

    print(f"Fetching database to {output_file}...")

    # Ensure attempts directory exists
    os.makedirs(attempts_dir, exist_ok=True)

    # Construct the command
    # We use explicit relative path for output to match the original command intent, 
    # but running it via subprocess with cwd set to pwa_dir is cleaner.
    command = [
        "npx", "wrangler", "d1", "export", "DB",
        "--remote",
        f"--output={os.path.abspath(output_file)}"
    ]

    try:
        # Run the command from the pwa directory
        subprocess.run(command, cwd=pwa_dir, check=True)
        print("\nDatabase downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading database: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'npx' command not found. Ensure Node.js and npm are installed.")
        sys.exit(1)

if __name__ == "__main__":
    fetch_database()
