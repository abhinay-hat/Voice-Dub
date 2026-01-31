"""
FFmpeg verification script.
Confirms ffmpeg-python wrapper can access FFmpeg binary and version is 4.0+.
"""
import sys
import subprocess
import re

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def verify_ffmpeg():
    """Verify FFmpeg installation and version."""
    print("Verifying FFmpeg installation...\n")

    # Test 1: Import ffmpeg-python
    try:
        import ffmpeg
        print("✓ ffmpeg-python library imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ffmpeg-python: {e}")
        return False

    # Test 2: Check FFmpeg binary accessibility
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        version_output = result.stdout
        print("✓ FFmpeg binary accessible from PATH")
    except FileNotFoundError:
        print("✗ FFmpeg binary not found in PATH")
        print("\nInstallation instructions:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("           or install via: choco install ffmpeg")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg command failed: {e}")
        return False

    # Test 3: Verify version is 4.0+
    version_match = re.search(r'ffmpeg version (\d+)\.(\d+)', version_output)
    if version_match:
        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        version_str = f"{major}.{minor}"
        print(f"✓ FFmpeg version: {version_str}")

        if major < 4:
            print(f"✗ FFmpeg version {version_str} is below required 4.0")
            print("   Please upgrade FFmpeg to version 4.0 or higher")
            return False
    else:
        print("⚠ Could not parse FFmpeg version, but binary is accessible")

    print("\n" + "="*60)
    print("FFmpeg verification PASSED ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    success = verify_ffmpeg()
    sys.exit(0 if success else 1)
