"""Test upload batch functionality to ensure samples are uploaded frequently."""
import unittest
import subprocess
import time
import os
import json
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset
import signal
import threading


class TestUploadBatchFunctionality(unittest.TestCase):
    """Test that upload batch size of 10 actually uploads every 10 samples."""
    
    def setUp(self):
        """Set up test environment."""
        self.api = HfApi()
        self.repo_name = "Thanarit/Thai-Voice-Test-1000000"
        self.project_root = Path(__file__).parent.parent
        self.main_sh_path = self.project_root / "main.sh"
        self.checkpoint_dir = self.project_root / "checkpoints"
        
        # Ensure main.sh is executable
        if self.main_sh_path.exists():
            os.chmod(self.main_sh_path, 0o755)
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up checkpoint files
        for checkpoint_file in self.checkpoint_dir.glob("*_unified_checkpoint.json"):
            checkpoint_file.unlink(missing_ok=True)
    
    def count_samples_on_huggingface(self, max_wait=30):
        """Count samples currently on HuggingFace, with retry logic."""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                dataset = load_dataset(self.repo_name, split="train", streaming=True)
                count = 0
                for _ in dataset:
                    count += 1
                return count
            except Exception as e:
                print(f"Error counting samples: {e}, retrying...")
                time.sleep(2)
        return 0
    
    def run_main_sh_with_timeout(self, args=None, timeout=60):
        """Run main.sh with a timeout and capture output."""
        cmd = [str(self.main_sh_path)]
        if args:
            cmd.extend(args)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.project_root),
            preexec_fn=os.setsid  # Create new process group for clean termination
        )
        
        # Start a timer to kill the process after timeout
        timer = threading.Timer(timeout, lambda: os.killpg(os.getpgid(process.pid), signal.SIGTERM))
        timer.start()
        
        try:
            stdout, stderr = process.communicate()
            timer.cancel()
            return process.returncode, stdout, stderr
        except:
            timer.cancel()
            raise
    
    def test_upload_happens_every_10_samples(self):
        """Test that uploads happen every 10 samples when batch size is 10."""
        print("\n=== Testing Upload Every 10 Samples ===")
        
        # Modify main.sh to use batch size 10 and only 50 samples for quick test
        main_sh_content = self.main_sh_path.read_text()
        test_content = main_sh_content.replace(
            'SAMPLES_PER_DATASET=1000000',
            'SAMPLES_PER_DATASET=50'
        ).replace(
            'UPLOAD_BATCH_SIZE=10',
            'UPLOAD_BATCH_SIZE=10'
        )
        
        # Create test version of main.sh
        test_main_sh = self.project_root / "test_main_batch.sh"
        test_main_sh.write_text(test_content)
        os.chmod(test_main_sh, 0o755)
        
        try:
            # Start fresh
            print("Starting fresh run with 50 samples...")
            returncode, stdout, stderr = self.run_main_sh_with_timeout(
                [str(test_main_sh)], 
                timeout=120
            )
            
            # Check samples were uploaded in batches
            print("Checking upload pattern in output...")
            upload_messages = [line for line in stdout.split('\n') if 'Uploaded shard' in line]
            
            # Should see multiple upload messages for 50 samples with batch size 10
            self.assertGreaterEqual(len(upload_messages), 4, 
                f"Expected at least 4 uploads for 50 samples with batch 10, got {len(upload_messages)}")
            
            # Verify samples on HuggingFace
            time.sleep(10)  # Wait for HF to process
            sample_count = self.count_samples_on_huggingface()
            self.assertGreaterEqual(sample_count, 40, 
                f"Expected at least 40 samples on HF, got {sample_count}")
            
        finally:
            # Cleanup
            test_main_sh.unlink(missing_ok=True)
    
    def test_upload_during_processing_not_just_at_end(self):
        """Test that uploads happen during processing, not just at the end."""
        print("\n=== Testing Uploads During Processing ===")
        
        # Create test script that processes 30 samples
        main_sh_content = self.main_sh_path.read_text()
        test_content = main_sh_content.replace(
            'SAMPLES_PER_DATASET=1000000',
            'SAMPLES_PER_DATASET=30'
        )
        
        test_main_sh = self.project_root / "test_main_batch_timing.sh"
        test_main_sh.write_text(test_content)
        os.chmod(test_main_sh, 0o755)
        
        try:
            # Run for 60 seconds then interrupt
            print("Running for 60 seconds to check intermediate uploads...")
            process = subprocess.Popen(
                [str(test_main_sh)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Monitor for upload messages
            upload_times = []
            start_time = time.time()
            
            # Read output for 60 seconds
            for _ in range(60):
                time.sleep(1)
                if process.poll() is not None:
                    break
                    
                # Check HuggingFace for samples
                current_count = self.count_samples_on_huggingface()
                if current_count > 0:
                    upload_times.append((time.time() - start_time, current_count))
                    print(f"  Found {current_count} samples after {time.time() - start_time:.1f}s")
            
            # Terminate process
            process.terminate()
            process.wait()
            
            # Verify we saw uploads during processing
            self.assertGreater(len(upload_times), 0, 
                "No uploads detected during processing - uploads may be buffered until end")
            
            # Verify uploads happened at regular intervals
            if len(upload_times) >= 2:
                # Check that samples increased by ~10 each time
                for i in range(1, len(upload_times)):
                    sample_increase = upload_times[i][1] - upload_times[i-1][1]
                    self.assertLessEqual(sample_increase, 15,
                        f"Sample increase of {sample_increase} suggests buffering larger than 10")
                        
        finally:
            test_main_sh.unlink(missing_ok=True)
    
    def test_interrupt_and_resume_with_small_batches(self):
        """Test interrupt and resume functionality with batch size 10."""
        print("\n=== Testing Interrupt and Resume with Batch Size 10 ===")
        
        # Create test script for 10000 samples
        main_sh_content = self.main_sh_path.read_text()
        test_content = main_sh_content.replace(
            'SAMPLES_PER_DATASET=1000000',
            'SAMPLES_PER_DATASET=10000'
        )
        
        test_main_sh = self.project_root / "test_main_interrupt.sh"
        test_main_sh.write_text(test_content)
        os.chmod(test_main_sh, 0o755)
        
        try:
            # Step 1: Run and interrupt after ~100 samples
            print("Step 1: Running with 10000 samples, will interrupt after ~100...")
            process = subprocess.Popen(
                [str(test_main_sh)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Wait for ~100 samples to be processed
            samples_before_interrupt = 0
            for i in range(120):  # Max 2 minutes
                time.sleep(1)
                
                # Check output for processing messages
                current_count = self.count_samples_on_huggingface()
                if current_count >= 100:
                    samples_before_interrupt = current_count
                    print(f"  Interrupting after {current_count} samples")
                    break
            
            # Interrupt the process
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            
            self.assertGreater(samples_before_interrupt, 0, 
                "No samples uploaded before interrupt")
            
            # Step 2: Resume processing
            print(f"\nStep 2: Resuming from {samples_before_interrupt} samples...")
            time.sleep(5)  # Wait for things to settle
            
            returncode, stdout, stderr = self.run_main_sh_with_timeout(
                [str(test_main_sh), "--resume"],
                timeout=300  # 5 minutes max
            )
            
            # Step 3: Verify resume worked correctly
            time.sleep(10)
            final_count = self.count_samples_on_huggingface()
            
            print(f"Final sample count: {final_count}")
            
            # Should have more samples than before
            self.assertGreater(final_count, samples_before_interrupt,
                f"Resume didn't add samples: before={samples_before_interrupt}, after={final_count}")
            
            # Check that IDs continued sequentially
            dataset = load_dataset(self.repo_name, split="train", streaming=True)
            sample_ids = []
            for sample in dataset:
                sample_ids.append(sample['ID'])
                if len(sample_ids) >= 20:
                    break
            
            # Verify sequential IDs
            for i in range(1, len(sample_ids)):
                prev_num = int(sample_ids[i-1][1:])
                curr_num = int(sample_ids[i][1:])
                self.assertEqual(curr_num, prev_num + 1,
                    f"Non-sequential IDs: {sample_ids[i-1]} -> {sample_ids[i]}")
                    
        finally:
            test_main_sh.unlink(missing_ok=True)
    
    def test_buffer_coordination(self):
        """Test that enhancement and speaker buffers don't block uploads."""
        print("\n=== Testing Buffer Coordination ===")
        
        # Check that batch sizes are synchronized in main.sh
        main_sh_content = self.main_sh_path.read_text()
        
        # Extract batch size settings
        self.assertIn('--speaker-batch-size $UPLOAD_BATCH_SIZE', main_sh_content,
            "Speaker batch size should use UPLOAD_BATCH_SIZE variable")
        self.assertIn('--enhancement-batch-size $UPLOAD_BATCH_SIZE', main_sh_content,
            "Enhancement batch size should use UPLOAD_BATCH_SIZE variable")
        self.assertIn('--upload-batch-size $UPLOAD_BATCH_SIZE', main_sh_content,
            "Upload batch size should use UPLOAD_BATCH_SIZE variable")
        
        # Verify UPLOAD_BATCH_SIZE is set to 10
        self.assertIn('UPLOAD_BATCH_SIZE=10', main_sh_content,
            "UPLOAD_BATCH_SIZE should be set to 10")


if __name__ == "__main__":
    unittest.main(verbosity=2)