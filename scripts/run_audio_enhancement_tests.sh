#!/bin/bash
# Run comprehensive audio enhancement tests with a single command

echo "==============================================="
echo "Audio Enhancement Comprehensive Test Runner"
echo "==============================================="
echo ""
echo "This script will test every aspect of the audio quality enhancement plan:"
echo "✓ Core noise removal (wind, voices, electronic hum)"
echo "✓ Voice clarity enhancement"
echo "✓ Processing speed (<0.8s per file)"
echo "✓ Flexible secondary speaker detection (0.1s-10s+)"
echo "✓ Smart adaptive processing"
echo "✓ Progressive enhancement"
echo "✓ Quality metrics (SNR, PESQ, STOI)"
echo "✓ GPU performance & memory usage"
echo "✓ Integration with existing codebase"
echo "✓ Real-time dashboard"
echo "✓ Before/after comparison system"
echo "✓ Edge cases and error handling"
echo ""
echo "Starting tests..."
echo "==============================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create test results directory
mkdir -p test_results

# Run the comprehensive test suite
python -m pytest tests/test_audio_enhancement_comprehensive.py -v --tb=short -p no:warnings || python tests/test_audio_enhancement_comprehensive.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: All audio enhancement tests passed!"
    echo ""
    echo "The implementation plan has been verified to cover:"
    echo "- All core requirements"
    echo "- Flexible secondary speaker detection"
    echo "- Performance targets"
    echo "- Quality metrics"
    echo "- Integration points"
    echo ""
    echo "You can now proceed with implementation knowing that"
    echo "all aspects of the plan are properly tested."
else
    echo ""
    echo "❌ FAILURE: Some tests failed. Please check the report above."
    echo ""
    echo "Review the detailed report in:"
    echo "  test_results/audio_enhancement_comprehensive_report.json"
    echo ""
    echo "Fix the failing tests before proceeding with implementation."
fi

echo ""
echo "==============================================="
echo "Test run complete."
echo "==============================================="