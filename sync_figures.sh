
SRC_DIR="./Figures/"
DEST_DIR="/home/vlongle/papers/ShELL-paper-2024/shell-cv-iclr/Figures/"


# Synchronize the content from SRC_DIR to DEST_DIR
rsync -avh --progress "$SRC_DIR" "$DEST_DIR"
echo "Synchronization complete."

# Navigate to the git directory
cd "/home/vlongle/papers/ShELL-paper-2024/"

# Check for changes and commit them if there are any
# Add changes to the staging area
git add shell-cv-iclr/Figures/

# Commit changes with a timestamped message
git commit -m "Updated Figures folder on $(date +'%Y-%m-%d %H:%M:%S')"

# Push changes to the remote repository
git push origin main
echo "Changes pushed to GitHub."

