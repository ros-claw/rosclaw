#!/bin/bash
# ROSClaw V4 Quick Start Script

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          ROSClaw V4 - Quick Start                          ║"
echo "║     Production-Ready Embodied Multi-Agent OS               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Python version:${NC} $PYTHON_VERSION"

# Install dependencies
echo
echo -e "${GREEN}▶ Installing ROSClaw V4...${NC}"
pip install -e ".[sim,vla,rl]" -q

# Check installation
echo
echo -e "${GREEN}▶ Verifying installation...${NC}"
python3 -c "import rosclaw_core; import rosclaw_sim; import rosclaw_mcp; import rosclaw_vla; import rosclaw_rl; print('All packages imported successfully')"

echo
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   Demo Options                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo -e "${YELLOW}1. End-to-End Demo (Full 7-Layer Pipeline)${NC}"
echo "   python -m rosclaw_vla.demo.e2e_demo --mode sim --task 'pick up the red cube'"
echo
echo -e "${YELLOW}2. SO101 Simulation${NC}"
echo "   python -m rosclaw_vla.demo.so101_sim"
echo
echo -e "${YELLOW}3. Conversation Interface${NC}"
echo "   python -m rosclaw_vla.demo.conversation_interface"
echo
echo -e "${YELLOW}4. List Available Robots${NC}"
echo "   python -c 'from rosclaw_core.builtins.so101 import SO101_BUILTIN; print(SO101_BUILTIN.name)'"
echo
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   Project Structure                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo
echo "rosclaw-v4/"
echo "├── src/"
echo "│   ├── rosclaw_core/     # Layer 1-2: Runtime & Data"
echo "│   ├── rosclaw_sim/      # Layer 3: Digital Twin (mjlab)"
echo "│   ├── rosclaw_mcp/      # Layer 4: MCP Servers (28 tools)"
echo "│   ├── rosclaw_vla/      # Layer 5: VLA Policy"
echo "│   └── rosclaw_rl/       # Layer 7: RL Training"
echo "├── configs/              # Robot configurations"
echo "└── docs/                 # Documentation"
echo
echo -e "${GREEN}✓ ROSClaw V4 Phase 1 is ready!${NC}"
echo
echo "Run the end-to-end demo now? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo
    python -m rosclaw_vla.demo.e2e_demo --mode sim --robot so101 --task "pick up the red cube"
fi
