#!/usr/bin/env python3
"""
Test CSV Export End-to-End Flow
Verifies that CSV exports work correctly from tool execution through API download
"""

import asyncio
import os
import requests
import time
from agent_sql_final_1212_patched import query_database, find_optimal_separation_conditions


async def test_query_export():
    """Test CSV export from query_database tool"""
    print("=" * 70)
    print("TEST 1: Query Database CSV Export")
    print("=" * 70)

    # Execute query with CSV export
    result = await query_database(
        sql_query="SELECT * FROM common_solvents_database WHERE polymer = 'PVDF' LIMIT 10",
        export_csv=True
    )

    print(result)

    # Extract export ID from result
    import re
    match = re.search(r'/api/export/([a-f0-9]{8})', result)

    if match:
        export_id = match.group(1)
        print(f"\n✅ Export ID found: {export_id}")

        # Verify export file exists
        export_dir = "./exports"
        if os.path.exists(export_dir):
            files = [f for f in os.listdir(export_dir) if export_id in f]
            if files:
                print(f"✅ Export file exists: {files[0]}")
                file_path = os.path.join(export_dir, files[0])

                # Check file size
                size = os.path.getsize(file_path)
                print(f"✅ File size: {size} bytes")

                # Read first few lines
                with open(file_path, 'r') as f:
                    lines = f.readlines()[:3]
                    print(f"✅ CSV preview (first 3 lines):")
                    for line in lines:
                        print(f"   {line.strip()}")

                return export_id
            else:
                print(f"❌ Export file not found in {export_dir}")
                return None
        else:
            print(f"❌ Export directory does not exist: {export_dir}")
            return None
    else:
        print("❌ No export ID found in result")
        return None


async def test_separation_export():
    """Test CSV export from separation analysis tool"""
    print("\n" + "=" * 70)
    print("TEST 2: Separation Analysis CSV Export")
    print("=" * 70)

    # Execute separation analysis with CSV export
    result = await find_optimal_separation_conditions(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature",
        solubility_column="solubility",
        target_polymer="PVDF",
        comparison_polymers="PET",
        start_temperature=25.0,
        initial_selectivity=30.0,
        export_csv=True
    )

    print(result)

    # Extract export ID from result
    import re
    match = re.search(r'/api/export/([a-f0-9]{8})', result)

    if match:
        export_id = match.group(1)
        print(f"\n✅ Export ID found: {export_id}")

        # Verify export file exists
        export_dir = "./exports"
        files = [f for f in os.listdir(export_dir) if export_id in f]
        if files:
            print(f"✅ Export file exists: {files[0]}")
            file_path = os.path.join(export_dir, files[0])

            # Check file contents
            with open(file_path, 'r') as f:
                lines = f.readlines()
                print(f"✅ CSV has {len(lines)} lines (including header)")
                print(f"✅ CSV preview (first 3 lines):")
                for line in lines[:3]:
                    print(f"   {line.strip()}")

            return export_id
        else:
            print(f"❌ Export file not found in {export_dir}")
            return None
    else:
        print("❌ No export ID found in result")
        return None


def test_api_endpoint(export_id):
    """Test downloading CSV via API endpoint"""
    print("\n" + "=" * 70)
    print("TEST 3: API Endpoint Download")
    print("=" * 70)

    url = f"http://localhost:8000/api/export/{export_id}"
    print(f"Requesting: {url}")

    try:
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            print(f"✅ HTTP Status: {response.status_code}")
            print(f"✅ Content-Type: {response.headers.get('Content-Type')}")
            print(f"✅ Content-Length: {len(response.content)} bytes")

            # Verify CSV content
            csv_content = response.text
            lines = csv_content.split('\n')
            print(f"✅ Downloaded CSV has {len(lines)} lines")
            print(f"✅ First line (header): {lines[0]}")

            return True
        else:
            print(f"❌ HTTP Status: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n🧪 CSV EXPORT END-TO-END TESTING\n")

    results = {
        "query_export": False,
        "separation_export": False,
        "api_download": False
    }

    # Test 1: Query database export
    export_id_1 = await test_query_export()
    results["query_export"] = export_id_1 is not None

    # Test 2: Separation analysis export
    export_id_2 = await test_separation_export()
    results["separation_export"] = export_id_2 is not None

    # Test 3: API endpoint (using first export ID)
    if export_id_1:
        results["api_download"] = test_api_endpoint(export_id_1)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! CSV export system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
