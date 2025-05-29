#!/usr/bin/env python3
"""
Test to verify speaker IDs are correctly assigned across datasets.
This test checks the actual output on HuggingFace.
"""

import unittest
from datasets import load_dataset
from collections import defaultdict


class TestSpeakerIdDatasetVerification(unittest.TestCase):
    """Verify speaker IDs in the actual HuggingFace dataset."""
    
    def test_speaker_id_requirements(self):
        """Test all speaker ID requirements are met."""
        print("\nLoading dataset from HuggingFace...")
        ds = load_dataset("Thanarit/Thai-Voice-Test6", split="train", streaming=True)
        
        # Collect samples by dataset
        samples_by_dataset = defaultdict(list)
        speaker_ids_by_dataset = defaultdict(set)
        all_samples = []
        
        for i, sample in enumerate(ds):
            if i >= 200:  # Process all 200 samples
                break
            
            all_samples.append(sample)
            dataset_name = sample['dataset_name']
            samples_by_dataset[dataset_name].append(sample)
            speaker_ids_by_dataset[dataset_name].add(sample['speaker_id'])
        
        # Test 1: Verify S1-S8,S10 have same speaker ID, S9 different
        print("\n1. Testing GigaSpeech2 speaker assignments...")
        gigaspeech_samples = {s['ID']: s['speaker_id'] for s in samples_by_dataset['GigaSpeech2'] if s['ID'] in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']}
        
        # S1-S8 and S10 should have the same speaker ID
        expected_same = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']
        speaker_ids_same = [gigaspeech_samples.get(sid) for sid in expected_same if sid in gigaspeech_samples]
        
        if speaker_ids_same:
            # All should be the same
            first_speaker = speaker_ids_same[0]
            for sid in speaker_ids_same:
                self.assertEqual(sid, first_speaker, 
                    f"Expected all of {expected_same} to have same speaker ID, but found different IDs")
            
            # S9 should be different
            if 'S9' in gigaspeech_samples:
                self.assertNotEqual(gigaspeech_samples['S9'], first_speaker,
                    f"S9 should have different speaker ID than {expected_same}, but has same ID: {gigaspeech_samples['S9']}")
            
            print(f"✓ S1-S8,S10 have speaker ID: {first_speaker}")
            print(f"✓ S9 has different speaker ID: {gigaspeech_samples.get('S9', 'N/A')}")
        
        # Test 2: Verify no speaker ID overlap between datasets
        print("\n2. Testing speaker ID separation between datasets...")
        datasets = list(speaker_ids_by_dataset.keys())
        print(f"Datasets found: {datasets}")
        
        for i, dataset1 in enumerate(datasets):
            for dataset2 in datasets[i+1:]:
                overlap = speaker_ids_by_dataset[dataset1] & speaker_ids_by_dataset[dataset2]
                self.assertEqual(len(overlap), 0,
                    f"Speaker IDs should not overlap between {dataset1} and {dataset2}, but found overlap: {overlap}")
        
        print("✓ No speaker ID overlap between datasets")
        
        # Test 3: Display speaker ID distribution
        print("\n3. Speaker ID distribution by dataset:")
        for dataset, speaker_ids in speaker_ids_by_dataset.items():
            print(f"\n{dataset}:")
            print(f"  Total samples: {len(samples_by_dataset[dataset])}")
            print(f"  Unique speakers: {len(speaker_ids)}")
            print(f"  Speaker IDs: {sorted(speaker_ids)}")
        
        # Test 4: Verify ProcessedVoiceTH doesn't start with SPK_00001
        print("\n4. Testing ProcessedVoiceTH speaker IDs...")
        if 'ProcessedVoiceTH' in speaker_ids_by_dataset:
            pv_speaker_ids = speaker_ids_by_dataset['ProcessedVoiceTH']
            self.assertNotIn('SPK_00001', pv_speaker_ids,
                "ProcessedVoiceTH should not contain SPK_00001 as it belongs to GigaSpeech2")
            self.assertNotIn('SPK_00002', pv_speaker_ids,
                "ProcessedVoiceTH should not contain SPK_00002 as it belongs to GigaSpeech2")
            
            # Get the minimum speaker ID number in ProcessedVoiceTH
            pv_speaker_numbers = [int(sid.split('_')[1]) for sid in pv_speaker_ids]
            min_pv_speaker = min(pv_speaker_numbers)
            
            # Get the maximum speaker ID number in GigaSpeech2
            if 'GigaSpeech2' in speaker_ids_by_dataset:
                gs_speaker_numbers = [int(sid.split('_')[1]) for sid in speaker_ids_by_dataset['GigaSpeech2']]
                max_gs_speaker = max(gs_speaker_numbers)
                
                self.assertGreater(min_pv_speaker, max_gs_speaker,
                    f"ProcessedVoiceTH should start after GigaSpeech2 speakers. "
                    f"GigaSpeech2 max: SPK_{max_gs_speaker:05d}, ProcessedVoiceTH min: SPK_{min_pv_speaker:05d}")
                
                print(f"✓ GigaSpeech2 speaker IDs end at: SPK_{max_gs_speaker:05d}")
                print(f"✓ ProcessedVoiceTH speaker IDs start at: SPK_{min_pv_speaker:05d}")
        
        print("\n✅ All speaker ID tests passed!")


if __name__ == "__main__":
    unittest.main()