#!/usr/bin/env python3
"""
Quick test script to verify all Phase 3 components
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import mlflow
        print("✅ MLflow imported successfully")
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        print("✅ Data science packages imported successfully")
    except ImportError as e:
        print(f"❌ Data science packages import failed: {e}")
        return False
    
    return True

def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing file existence...")
    
    required_files = [
        'data/International_Education_Costs.csv',
        'streamlit_app.py',
        'api.py', 
        'model_registry.py',
        'monitoring.py',
        'deployment.sh',
        'requirements.txt',
        '.gitignore'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_exist = False
    
    return all_files_exist

def test_data_loading():
    """Test if data can be loaded correctly"""
    print("\n📊 Testing data loading...")
    
    try:
        import pandas as pd
        df = pd.read_csv('data/International_Education_Costs.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check required columns
        required_cols = ['Country', 'Program', 'Level', 'Tuition_USD', 'Rent_USD']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            return False
        else:
            print("✅ All required columns present")
            return True
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_components():
    """Test if model components can be imported"""
    print("\n🤖 Testing model components...")
    
    try:
        # Test TCAPredictor import
        spec = importlib.util.spec_from_file_location("model_registry", "model_registry.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        predictor = model_module.TCAPredictor()
        print("✅ TCAPredictor class imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 EduSpend Phase 3 Component Test")
    print("=" * 50)
    
    # Run tests
    import_test = test_imports()
    file_test = test_files()
    data_test = test_data_loading()
    model_test = test_model_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    tests = [
        ("Package Imports", import_test),
        ("File Existence", file_test),
        ("Data Loading", data_test),
        ("Model Components", model_test)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 3 deployment is ready!")
        print("\n🚀 Quick Start Commands:")
        print("  Streamlit: streamlit run streamlit_app.py")
        print("  API:       python api.py")
        print("  MLflow:    mlflow ui --backend-store-uri ./mlruns")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
