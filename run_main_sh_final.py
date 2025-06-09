"""
Run main.sh as final verification
"""
import subprocess
import os

def run_main_sh():
    """Run main.sh with default settings"""
    print("Running ./main.sh as final verification...")
    print("This will process 100 samples with ultra_aggressive enhancement")
    print("=" * 60)
    
    # Run main.sh
    cmd = ["./main.sh"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Track interesting events
    s5_lines = []
    intelligent_lines = []
    secondary_lines = []
    upload_lines = []
    
    for line in iter(process.stdout.readline, ''):
        if line:
            # Print everything
            print(line.rstrip())
            
            # Collect specific lines
            if 'S5' in line or 'train_5' in line:
                s5_lines.append(line.strip())
            
            if 'intelligent' in line.lower():
                intelligent_lines.append(line.strip())
            
            if 'secondary' in line.lower() and ('removal' in line.lower() or 'removed' in line.lower()):
                secondary_lines.append(line.strip())
            
            if 'upload' in line.lower() or 'huggingface' in line.lower():
                upload_lines.append(line.strip())
    
    process.wait()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nS5 mentions: {len(s5_lines)}")
    for line in s5_lines[-5:]:  # Last 5
        print(f"  - {line}")
    
    print(f"\nIntelligent silencer mentions: {len(intelligent_lines)}")
    for line in intelligent_lines:
        print(f"  - {line}")
    
    print(f"\nSecondary removal mentions: {len(secondary_lines)}")
    for line in secondary_lines[-5:]:  # Last 5
        print(f"  - {line}")
    
    print(f"\nUpload mentions: {len(upload_lines)}")
    for line in upload_lines[-5:]:  # Last 5
        print(f"  - {line}")
    
    return process.returncode

if __name__ == "__main__":
    exit_code = run_main_sh()
    
    print(f"\nExit code: {exit_code}")
    
    if exit_code == 0:
        print("\n✅ main.sh completed successfully")
        print("\nCheck https://huggingface.co/datasets/Thanarit/Thai-Voice-10000000 for the uploaded dataset")
        print("Look for S5 in the dataset viewer and check if the secondary speaker at the end has been removed")
    else:
        print("\n❌ main.sh failed")