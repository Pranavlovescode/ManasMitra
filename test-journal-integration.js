/**
 * Test script to verify journal analysis integration
 * Run with: node test-journal-integration.js
 */

const testJournalAnalysis = async () => {
  console.log('🧠 Testing Journal Analysis Integration...\n');

  // Test 1: Direct FastAPI endpoint
  console.log('1️⃣ Testing FastAPI endpoint directly...');
  try {
    const response = await fetch('http://127.0.0.1:8000/analyze/journal', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        title: "Feeling overwhelmed today",
        content: "I've been feeling really anxious about work lately. Everything seems impossible and I keep thinking the worst will happen. I feel like I'm failing at everything and nothing I do is good enough.",
        mood: "sad",
        prompt: "How are you feeling today?"
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ FastAPI analysis successful!');
      console.log(`   - Overall sentiment: ${data.overall_sentiment}`);
      console.log(`   - Key themes: ${data.key_themes.join(', ')}`);
      console.log(`   - Detected emotion: ${data.content_analysis.emotion}`);
      console.log(`   - Risk score: ${data.content_analysis.risk_score}`);
      console.log(`   - Distortions: ${data.content_analysis.distortions.join(', ')}`);
      console.log(`   - Recommendations: ${data.recommendations.length} suggestions`);
    } else {
      console.log(`❌ FastAPI endpoint failed: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    console.log(`❌ FastAPI connection error: ${error.message}`);
    console.log('   Make sure FastAPI server is running on http://127.0.0.1:8000');
  }

  console.log('\n' + '='.repeat(50));

  // Test 2: Health check
  console.log('2️⃣ Testing FastAPI health...');
  try {
    const healthResponse = await fetch('http://127.0.0.1:8000/health');
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ FastAPI health check passed!');
      console.log(`   - Status: ${healthData.status}`);
      console.log(`   - Models loaded:`, healthData.models_loaded);
    } else {
      console.log(`❌ Health check failed: ${healthResponse.status}`);
    }
  } catch (error) {
    console.log(`❌ Health check error: ${error.message}`);
  }

  console.log('\n' + '='.repeat(50));

  // Test 3: Model status
  console.log('3️⃣ Testing model availability...');
  try {
    const modelResponse = await fetch('http://127.0.0.1:8000/models/status');
    if (modelResponse.ok) {
      const modelData = await modelResponse.json();
      console.log('✅ Model status check passed!');
      console.log(`   - Status: ${modelData.status_message}`);
      console.log(`   - Intent model: ${modelData.intent ? '✓' : '✗'}`);
      console.log(`   - Emotion model: ${modelData.emotion ? '✓' : '✗'}`);
      console.log(`   - Cognitive model: ${modelData.cognitive ? '✓' : '✗'}`);
      console.log(`   - Risk model: ${modelData.risk ? '✓' : '✗'}`);
    } else {
      console.log(`❌ Model status failed: ${modelResponse.status}`);
    }
  } catch (error) {
    console.log(`❌ Model status error: ${error.message}`);
  }

  console.log('\n🎯 Integration Test Summary:');
  console.log('   - FastAPI server should be running on http://127.0.0.1:8000');
  console.log('   - Next.js API routes are configured to call FastAPI');
  console.log('   - Journal model updated to store analysis results');
  console.log('   - Use POST /api/journal with journal data to test full integration');
};

// Run if called directly
if (typeof window === 'undefined') {
  testJournalAnalysis().catch(console.error);
}

module.exports = { testJournalAnalysis };