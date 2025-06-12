#!/bin/bash

echo "=== DeepFilterNet Installation Script ==="
echo

# Step 1: Install Rust
echo "Step 1: Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify Rust installation
echo "Rust version:"
rustc --version
cargo --version
echo

# Step 2: Install system dependencies
echo "Step 2: Installing system dependencies..."
echo "Note: You may need to run these with sudo:"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y pkg-config libssl-dev"
echo

# Step 3: Install DeepFilterNet
echo "Step 3: Installing DeepFilterNet..."
pip install deepfilternet

# Step 4: Verify installation
echo
echo "Step 4: Verifying installation..."
python -c "from df import enhance, init_df; print('✓ DeepFilterNet successfully installed!')"

echo
echo "=== Installation Complete ==="
echo
echo "To use DeepFilterNet in your code:"
echo "  from df import enhance, init_df"
echo "  model, df_state, _ = init_df()"
echo "  enhanced_audio = enhance(model, df_state, noisy_audio)"