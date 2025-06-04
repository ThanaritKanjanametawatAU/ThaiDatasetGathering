#!/usr/bin/env python3
"""
Test to verify HuggingFace dataset has correct transcripts and speaker clustering.
This test connects to the actual HuggingFace dataset and verifies the data.
"""

import unittest
from datasets import load_dataset
import time


class TestHuggingFaceVerification(unittest.TestCase):
    """Test that verifies the actual data on HuggingFace."""
    
    def setUp(self):
        """Set up test environment."""
        self.dataset_name = "Thanarit/Thai-Voice"
        self.expected_speaker_group = "SPK_00001"  # S1-S8 and S10 should have this
        self.expected_different_speaker = "SPK_00002"  # S9 should have this
        
    def test_dataset_has_transcripts_and_correct_speaker_ids(self):
        """Test that all samples have transcripts and correct speaker clustering."""
        print(f"\n{'='*60}")
        print("VERIFYING HUGGINGFACE DATASET")
        print(f"{'='*60}")
        
        # Give HuggingFace time to process the upload
        print("Waiting 5 seconds for HuggingFace to process...")
        time.sleep(5)
        
        try:
            # Load the dataset from HuggingFace
            print(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split='train', streaming=True)
            
            # Take first 20 samples to verify (includes S1-S10 from GigaSpeech2)
            samples = list(dataset.take(20))
            
            print(f"\nFound {len(samples)} samples")
            print("\nVerifying each sample:")
            print(f"{'ID':<10} {'Speaker ID':<15} {'Has Transcript':<15} {'Dataset':<20}")
            print("-" * 60)
            
            # Track findings
            missing_transcripts = []
            speaker_id_issues = []
            
            for sample in samples:
                sample_id = sample['ID']
                speaker_id = sample.get('speaker_id', 'MISSING')
                transcript = sample.get('transcript', '')
                dataset_name = sample.get('dataset_name', 'Unknown')
                
                has_transcript = bool(transcript and transcript.strip())
                
                print(f"{sample_id:<10} {speaker_id:<15} {str(has_transcript):<15} {dataset_name:<20}")
                
                # Check transcript
                if not has_transcript:
                    missing_transcripts.append(sample_id)
                
                # Check speaker ID clustering for GigaSpeech2 samples
                if dataset_name == "GigaSpeech2":
                    if sample_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']:
                        if speaker_id != self.expected_speaker_group:
                            speaker_id_issues.append(f"{sample_id} has {speaker_id}, expected {self.expected_speaker_group}")
                    elif sample_id == 'S9':
                        if speaker_id != self.expected_different_speaker:
                            speaker_id_issues.append(f"{sample_id} has {speaker_id}, expected {self.expected_different_speaker}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("VERIFICATION SUMMARY:")
            print(f"{'='*60}")
            
            if missing_transcripts:
                print(f"\n❌ FAILED: Missing transcripts for: {', '.join(missing_transcripts)}")
            else:
                print("\n✅ PASSED: All samples have transcripts")
            
            if speaker_id_issues:
                print(f"\n❌ FAILED: Speaker ID issues:")
                for issue in speaker_id_issues:
                    print(f"   - {issue}")
            else:
                print("\n✅ PASSED: Speaker clustering is correct")
                print(f"   - S1-S8, S10 have {self.expected_speaker_group}")
                print(f"   - S9 has {self.expected_different_speaker}")
            
            # Assertions
            self.assertEqual(len(missing_transcripts), 0, 
                           f"Samples missing transcripts: {missing_transcripts}")
            self.assertEqual(len(speaker_id_issues), 0,
                           f"Speaker ID clustering issues: {speaker_id_issues}")
            
            print(f"\n{'='*60}")
            print("ALL VERIFICATIONS PASSED! ✅")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ ERROR loading dataset: {str(e)}")
            print("This might happen if the dataset was just uploaded. Try again in a few seconds.")
            raise

    def test_dataset_sample_count(self):
        """Test that the dataset has the expected number of samples."""
        print(f"\n{'='*60}")
        print("CHECKING SAMPLE COUNT")
        print(f"{'='*60}")
        
        try:
            # Load dataset info
            dataset = load_dataset(self.dataset_name, split='train', streaming=True)
            
            # Count samples (for streaming, we need to iterate)
            count = 0
            for _ in dataset:
                count += 1
                if count % 50 == 0:
                    print(f"Counted {count} samples so far...")
            
            print(f"\nTotal samples in dataset: {count}")
            
            # For the full test with resume, we expect 200 samples
            # For smaller tests, we might have fewer
            self.assertGreater(count, 0, "Dataset should have at least some samples")
            
            print(f"\n✅ Dataset has {count} samples")
            
        except Exception as e:
            print(f"\n❌ ERROR counting samples: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main(verbosity=2)