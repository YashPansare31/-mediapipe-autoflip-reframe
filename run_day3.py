import subprocess
import sys
import os

def run_all_day3_tests():
    """Run comprehensive Day 3 tests"""
    
    print("=== Day 3 Complete Test Suite ===")
    
    tests = [
        ("Setup Check", "python setup_autoflip.py"),
        ("Performance Test", "python src/optimization/performance.py"),
        ("Encoding Test", "python src/encoding/video_encoder.py"),
        ("Edge Cases Test", "python src/edge_cases/handler.py"),
        ("Production Pipeline", "python test_day3_complete.py --test"),
    ]
    
    results = []
    
    for test_name, command in tests:
        print(f"\n--- Running {test_name} ---")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"ok {test_name} PASSED")
                results.append((test_name, True))
            else:
                print(f"X {test_name} FAILED")
                print(f"Error: {result.stderr}")
                results.append((test_name, False))
        except subprocess.TimeoutExpired:
            print(f"ERR {test_name} TIMEOUT")
            results.append((test_name, False))
        except Exception as e:
            print(f"X {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n=== Day 3 Test Results ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n Day 3 COMPLETE! Ready for Day 4 (Polish & Documentation)")
    else:
        print(f"\nERR {total-passed} tests need attention before Day 4")
    
    return passed == total

if __name__ == "__main__":
    run_all_day3_tests()