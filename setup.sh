#!/bin/bash

# Quick Start Setup for React + shadcn/ui Mural
# Run this after extracting all files to your project directory

echo "🎨 Crown Mural - React Migration Setup"
echo "======================================"
echo ""

# Check Node version
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ required. You have: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"
echo ""

# Install npm dependencies
echo "📦 Installing npm dependencies..."
npm install --legacy-peer-deps

if [ $? -ne 0 ]; then
    echo "❌ npm install failed. Trying without legacy flag..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed. Please check errors above."
        exit 1
    fi
fi

echo "✅ Dependencies installed"
echo ""

# Check for Python (optional)
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 detected: $(python3 --version)"
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing Python dependencies..."
        python3 -m pip install -r requirements.txt --quiet
        
        if [ $? -eq 0 ]; then
            echo "✅ Python dependencies installed"
        else
            echo "⚠️  Python dependencies installation failed (non-critical)"
        fi
    fi
else
    echo "⚠️  Python 3 not found. Image processing won't work."
    echo "   You can skip this if you already have processed images."
fi

echo ""

# Check if mural image exists
if [ -f "Mural_Crown_of_Italian_City.svg.png" ]; then
    echo "✅ Source mural image found"
else
    echo "⚠️  Source image not found: Mural_Crown_of_Italian_City.svg.png"
    echo "   Place your mural image in the root directory and run:"
    echo "   npm run process:image"
fi

echo ""

# Check if data directory exists
if [ -d "public/data" ] && [ -f "public/data/metadata.json" ]; then
    echo "✅ Processed data found in public/data/"
else
    echo "⚠️  Processed data not found. You need to run:"
    echo "   npm run process:image"
    echo "   (Requires Python and source image)"
fi

echo ""
echo "======================================"
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "  1. Ensure you have processed image data:"
echo "     npm run process:image"
echo ""
echo "  2. Start development server:"
echo "     npm run dev"
echo ""
echo "  3. Build for production:"
echo "     npm run build"
echo ""
echo "  4. Preview production build:"
echo "     npm run preview"
echo ""
echo "📚 See README.md for more details"
echo "🔧 See MIGRATION.md for upgrade notes"
echo "📋 See FILE_CHANGES.md for file mapping"
echo ""
echo "Happy coding! 🚀"
