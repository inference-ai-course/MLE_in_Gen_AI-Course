#!/usr/bin/env python3
"""
Setup script for AI Safety Demonstrations
Helps users quickly set up the environment and run demos
"""

import os
import sys
import subprocess
import platform


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ pip not found. Please install pip first.")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found. Please run from class8 directory.")
        return False


def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️ OpenAI API key not found")
        print("To use real API features, set: export OPENAI_API_KEY='your-key-here'")
        return False


def run_basic_demo():
    """Run a basic demonstration"""
    print("\n🚀 Running basic demonstration...")
    
    try:
        result = subprocess.run([sys.executable, "demo_runner.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            print("\nDemo output (last 10 lines):")
            for line in lines[-10:]:
                print(f"  {line}")
            return True
        else:
            print(f"❌ Demo failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Demo timed out")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False


def show_usage_instructions():
    """Show instructions for using the demos"""
    print("\n📚 USAGE INSTRUCTIONS")
    print("=" * 25)
    
    print("\n1. Basic Demonstrations:")
    print("   python ai_safety_demos.py          # Core safety demo")
    print("   python demo_runner.py              # Comprehensive tests")
    
    print("\n2. Interactive Learning:")
    print("   jupyter notebook interactive_ai_safety_demo.ipynb")
    
    print("\n3. API Integration (requires OpenAI key):")
    print("   export OPENAI_API_KEY='your-key'")
    print("   python chatgpt_api_safety_demo.py")
    
    print("\n4. Testing:")
    print("   python -m pytest test_ai_safety_demos.py -v")
    
    print("\n📖 Files Overview:")
    files = [
        ("ai_safety_demos.py", "Core safety demonstration modules"),
        ("test_ai_safety_demos.py", "Comprehensive test suite"),
        ("chatgpt_api_safety_demo.py", "Real-world API integration"),
        ("interactive_ai_safety_demo.ipynb", "Interactive Jupyter notebook"),
        ("demo_runner.py", "Simple demo runner"),
        ("requirements.txt", "Package dependencies"),
        ("README.md", "Detailed documentation")
    ]
    
    for filename, description in files:
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"   {status} {filename:30} - {description}")


def main():
    """Main setup function"""
    print("🛡️ AI SAFETY DEMONSTRATIONS SETUP")
    print("=" * 40)
    print("Welcome to the AI Safety educational toolkit!")
    print("This setup will help you get started with demonstrations of:")
    print("• AI Hallucinations")
    print("• Jailbreak Techniques") 
    print("• Bias and Ethical Issues")
    print("• Production Safety Measures")
    
    # Check system requirements
    print(f"\n🔍 System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    install_success = install_requirements()
    
    # Check OpenAI API key
    has_api_key = check_openai_key()
    
    # Run basic demo if installation succeeded
    if install_success:
        demo_success = run_basic_demo()
    else:
        demo_success = False
    
    # Show usage instructions
    show_usage_instructions()
    
    # Final status
    print(f"\n🎯 SETUP SUMMARY")
    print("=" * 20)
    print(f"✅ Python version: Compatible")
    print(f"{'✅' if install_success else '❌'} Package installation: {'Success' if install_success else 'Failed'}")
    print(f"{'✅' if has_api_key else '⚠️'} OpenAI API key: {'Found' if has_api_key else 'Not set (optional)'}")
    print(f"{'✅' if demo_success else '❌'} Basic demo: {'Working' if demo_success else 'Failed'}")
    
    if install_success and demo_success:
        print(f"\n🎉 Setup complete! You're ready to explore AI safety.")
        print(f"Start with: python demo_runner.py")
        if not has_api_key:
            print(f"💡 For full API features, set your OpenAI API key")
    else:
        print(f"\n⚠️ Setup had some issues. Check error messages above.")
        if not install_success:
            print("Try manually installing: pip install -r requirements.txt")
    
    print(f"\n📖 For detailed documentation, see README.md")
    print(f"🛡️ Remember: These tools are for educational purposes only!")


if __name__ == "__main__":
    main() 