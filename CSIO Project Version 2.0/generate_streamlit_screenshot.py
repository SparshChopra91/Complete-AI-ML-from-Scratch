"""
Generate Streamlit Dashboard Screenshot for Figure 9.4
This script starts the Streamlit app and captures a screenshot using Chrome.
"""

import subprocess
import time
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_streamlit_screenshot():
    """
    Figure 9.4: Streamlit Dashboard Screenshot
    Starts Streamlit app and captures a screenshot using Chrome headless mode.
    """
    print("Generating Figure 9.4: Streamlit Dashboard Screenshot...")
    
    # Output path
    output_path = OUTPUT_DIR / "figure_9_4_streamlit_dashboard.png"
    
    # Start Streamlit in background
    print("Starting Streamlit app...")
    streamlit_process = subprocess.Popen(
        ["python", "-m", "streamlit", "run", "app.py", 
         "--server.port", "8501", 
         "--server.headless", "true"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for Streamlit to start
    print("Waiting for Streamlit to start...")
    time.sleep(10)  # Give it time to start
    
    try:
        # Use Chrome in headless mode to capture screenshot
        print("Capturing screenshot with Chrome...")
        
        # Chrome headless screenshot command
        chrome_cmd = [
            "chrome",
            "--headless",
            "--disable-gpu",
            "--screenshot=" + str(output_path),
            "--window-size=1920,1080",
            "--hide-scrollbars",
            "http://localhost:8501"
        ]
        
        # Try to find Chrome executable
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        
        chrome_found = False
        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                chrome_cmd[0] = chrome_path
                chrome_found = True
                break
        
        if not chrome_found:
            # Try using 'chrome' command directly
            print("Chrome not found in standard paths, trying 'chrome' command...")
        
        # Run Chrome to capture screenshot
        result = subprocess.run(chrome_cmd, capture_output=True, text=True, timeout=30)
        
        if output_path.exists():
            print("[OK] Saved Streamlit Dashboard Screenshot to: " + str(output_path))
            return output_path
        else:
            print("[WARNING] Screenshot may not have been captured. Checking alternative...")
            
            # Alternative: Create a placeholder screenshot with instructions
            create_placeholder_screenshot(output_path)
            return output_path
            
    except subprocess.TimeoutExpired:
        print("[WARNING] Chrome timed out. Creating placeholder screenshot...")
        create_placeholder_screenshot(output_path)
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Error capturing screenshot: {e}")
        print("Creating placeholder screenshot...")
        create_placeholder_screenshot(output_path)
        return output_path
        
    finally:
        # Stop Streamlit
        print("Stopping Streamlit app...")
        streamlit_process.terminate()
        streamlit_process.wait()

def create_placeholder_screenshot(output_path):
    """Create a placeholder screenshot if automated capture fails."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#fffaf2')
    ax.set_facecolor('#ffffff')
    
    # Title
    ax.text(0.5, 0.85, 'Smart Clinic Assistant', fontsize=32, fontweight='bold',
            ha='center', va='center', color='#28313a', transform=ax.transAxes)
    
    ax.text(0.5, 0.75, 'Streamlit Clinical Decision Support Dashboard', fontsize=18,
            ha='center', va='center', color='#746c62', transform=ax.transAxes)
    
    # Add description
    description = (
        "Two-tier heart disease triage system with:\n"
        "• Tier 1: Basic vitals intake (Age, Sex, Chest Pain, Blood Pressure)\n"
        "• Tier 2: Full diagnostic panel with SHAP explanations\n"
        "• Real-time risk assessment and clinical decision support"
    )
    
    ax.text(0.5, 0.45, description, fontsize=14,
            ha='center', va='center', color='#28313a', 
            transform=ax.transAxes, linespacing=1.8,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff8ef', 
                     edgecolor='#eadfce', alpha=0.9))
    
    # Footer
    ax.text(0.5, 0.1, 'To capture live dashboard: streamlit run app.py', fontsize=12,
            ha='center', va='center', color='#9c9286', transform=ax.transAxes,
            style='italic')
    
    ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("[OK] Created placeholder screenshot at: " + str(output_path))

def main():
    """Main function to generate Streamlit dashboard screenshot."""
    print("=" * 60)
    print("Smart Clinic Assistant - Streamlit Dashboard Screenshot")
    print("=" * 60)
    print()
    
    # Generate screenshot
    screenshot_path = generate_streamlit_screenshot()
    
    print()
    print("=" * 60)
    print("Streamlit Dashboard Screenshot Complete!")
    print("=" * 60)
    print("Output: " + str(screenshot_path))
    print()
    print("Use this as Figure 9.4 in your report.")
    print()
    print("NOTE: For a live dashboard screenshot, run:")
    print("  streamlit run app.py")
    print("  Then manually take a screenshot from your browser.")

if __name__ == "__main__":
    main()
