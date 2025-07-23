#!/usr/bin/env python3
"""Complete Jaeger debugging tool to diagnose trace export issues."""

import time
import requests
import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter


def check_jaeger_services():
    """Check what services Jaeger currently knows about."""
    print("🔍 Checking existing services in Jaeger...")
    
    try:
        response = requests.get("http://localhost:16686/api/services", timeout=10)
        if response.status_code == 200:
            services = response.json()
            print(f"✅ Found {len(services['data'])} services in Jaeger:")
            for service in services['data']:
                print(f"   - {service}")
            return services['data']
        else:
            print(f"❌ Failed to get services: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking services: {e}")
    
    return []


def check_jaeger_health():
    """Check Jaeger collector health."""
    print("🏥 Checking Jaeger collector health...")
    
    # Check UI
    try:
        response = requests.get("http://localhost:16686", timeout=5)
        if response.status_code == 200:
            print("✅ Jaeger UI is accessible")
        else:
            print(f"⚠️ Jaeger UI returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Jaeger UI not accessible: {e}")
    
    # Check collector endpoint
    try:
        # POST a small test to see if collector responds properly
        test_data = {"test": "connectivity"}
        response = requests.post(
            "http://localhost:14268/api/traces", 
            json=test_data, 
            timeout=5,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Jaeger collector responded (status: {response.status_code})")
    except requests.exceptions.ConnectTimeout:
        print("❌ Jaeger collector timeout")
    except requests.exceptions.ConnectionError:
        print("❌ Jaeger collector connection refused")
    except Exception as e:
        print(f"⚠️ Jaeger collector response: {e}")


def test_direct_trace_export():
    """Test direct trace export to Jaeger with verbose logging."""
    print("\n🧪 Testing direct trace export...")
    
    # Create resource with explicit service name
    resource = Resource.create({
        "service.name": "debug-trace-test",
        "service.version": "1.0.0",
        "service.instance.id": "debug-instance"
    })
    
    print(f"📋 Resource created: {resource.attributes}")
    
    # Setup tracer with console export for debugging
    tracer_provider = TracerProvider(resource=resource)
    
    # Add console exporter to see what spans are generated
    console_exporter = ConsoleSpanExporter()
    console_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(console_processor)
    
    # Add Jaeger exporter
    try:
        jaeger_exporter = JaegerExporter(
            collector_endpoint="http://localhost:14268/api/traces",
        )
        jaeger_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(jaeger_processor)
        print("✅ Jaeger exporter configured")
    except Exception as e:
        print(f"❌ Failed to configure Jaeger exporter: {e}")
        return
    
    trace.set_tracer_provider(tracer_provider)
    
    # Create test trace
    tracer = trace.get_tracer("debug-tracer", "1.0.0")
    
    print("🎯 Generating test span...")
    with tracer.start_as_current_span("debug_test_span") as span:
        span.set_attribute("test.type", "connectivity_debug")
        span.set_attribute("test.timestamp", str(time.time()))
        span.set_attribute("service.name", "debug-trace-test")
        span.add_event("Debug test started")
        
        # Get span details
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        
        print(f"📊 Generated trace ID: {trace_id}")
        print(f"📊 Generated span ID: {span_id}")
        
        # Simulate some work
        time.sleep(0.1)
        span.add_event("Debug test completed")
    
    # Force export and wait
    print("🚀 Forcing span export...")
    try:
        jaeger_processor.force_flush(timeout_millis=10000)
        console_processor.force_flush(timeout_millis=5000)
        print("✅ Force flush completed")
    except Exception as e:
        print(f"⚠️ Force flush error: {e}")
    
    # Wait a bit for async export
    print("⏳ Waiting for async export...")
    time.sleep(3)
    
    return trace_id


def search_for_trace(trace_id):
    """Search for a specific trace in Jaeger."""
    print(f"\n🔍 Searching for trace {trace_id} in Jaeger...")
    
    try:
        # Search by trace ID
        url = f"http://localhost:16686/api/traces/{trace_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            trace_data = response.json()
            if trace_data.get('data') and len(trace_data['data']) > 0:
                print("✅ Trace found in Jaeger!")
                trace_info = trace_data['data'][0]
                print(f"   Service: {trace_info.get('processes', {}).values()}")
                print(f"   Spans: {len(trace_info.get('spans', []))}")
                return True
            else:
                print("❌ Trace not found in Jaeger")
        else:
            print(f"❌ Trace search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error searching for trace: {e}")
    
    return False


def test_your_app_traces():
    """Test your actual application's trace endpoints."""
    print("\n🧪 Testing your application's trace generation...")
    
    endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/health/",
        "http://localhost:8000/health/test-jaeger"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"📡 Testing {endpoint}")
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ Response: {response.status_code}")
                
                # Check for trace ID in response
                if 'trace_id' in response.text:
                    data = response.json()
                    if 'trace_id' in data:
                        print(f"   📊 Trace ID: {data['trace_id']}")
                        return data['trace_id']
                
                # Check for trace ID in headers
                if 'X-Trace-ID' in response.headers:
                    print(f"   📊 Trace ID (header): {response.headers['X-Trace-ID']}")
                    return response.headers['X-Trace-ID']
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None


def diagnose_service_name_issue():
    """Diagnose service name configuration issues."""
    print("\n🔧 Diagnosing service name configuration...")
    
    # Check your app's service name
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"📋 Your app reports service name: {data.get('service_name', 'NOT FOUND')}")
            print(f"📋 Jaeger endpoint: {data.get('jaeger_endpoint', 'NOT FOUND')}")
        else:
            print("❌ Cannot get app configuration")
    except Exception as e:
        print(f"❌ Error getting app config: {e}")


def main():
    """Run complete Jaeger debugging."""
    print("🚀 JAEGER DEBUGGING TOOL")
    print("=" * 50)
    
    # Step 1: Check Jaeger health
    check_jaeger_health()
    print()
    
    # Step 2: Check existing services
    existing_services = check_jaeger_services()
    print()
    
    # Step 3: Diagnose your app configuration
    diagnose_service_name_issue()
    print()
    
    # Step 4: Test direct trace export
    trace_id = test_direct_trace_export()
    print()
    
    # Step 5: Search for the test trace
    if trace_id:
        found = search_for_trace(trace_id)
        if found:
            print("✅ Direct trace export is working!")
        else:
            print("❌ Direct trace export failed - trace not found")
    print()
    
    # Step 6: Test your application traces
    app_trace_id = test_your_app_traces()
    if app_trace_id:
        time.sleep(2)  # Wait for export
        found = search_for_trace(app_trace_id)
        if found:
            print("✅ Your application traces are working!")
        else:
            print("❌ Your application traces are not reaching Jaeger")
    print()
    
    # Final recommendations
    print("🔧 RECOMMENDATIONS:")
    print("=" * 50)
    
    if not existing_services:
        print("❌ No services found in Jaeger - this indicates a major connectivity issue")
        print("   1. Ensure Jaeger is running: docker ps | grep jaeger")
        print("   2. Check port mapping: docker port jaeger")
        print("   3. Try restarting Jaeger: docker-compose down && docker-compose up -d")
    
    if 'evaluation-service' not in existing_services:
        print("❌ Your service 'evaluation-service' not found in Jaeger")
        print("   1. Check service name consistency in your code")
        print("   2. Verify spans are being exported")
        print("   3. Check for span export errors in your app logs")
    
    print("\n📊 Check Jaeger UI: http://localhost:16686")
    print("🔍 Look for services: debug-trace-test, evaluation-service")


if __name__ == "__main__":
    main()