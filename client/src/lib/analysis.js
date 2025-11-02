// Lightweight journal trends analysis utilities
// Aggregates mood, sentiment, risk, and cognitive distortion trends

export function computeJournalTrends(journals = []) {
  if (!Array.isArray(journals)) return {};

  const byDate = (d) => new Date(d).toISOString().slice(0, 10);

  // Map moods to a numeric score for simple trend lines
  const moodMap = { sad: 1, neutral: 3, happy: 4, excited: 5, loved: 5 };
  const sentimentMap = { negative: 1, neutral: 3, positive: 5 };
  const riskMap = { low: 1, medium: 3, high: 5 };

  const moodsOverTime = [];
  const sentimentTrend = [];
  const riskTrend = [];
  const distortionCount = {};
  const dateCounts = {};

  let firstDate = null;
  let lastDate = null;

  for (const j of journals) {
    const dt = byDate(j.createdAt || j.updatedAt || Date.now());
    if (!firstDate || dt < firstDate) firstDate = dt;
    if (!lastDate || dt > lastDate) lastDate = dt;

    // Mood trend
    const mood = (j.mood || 'neutral').toLowerCase();
    moodsOverTime.push({ date: dt, mood, moodScore: moodMap[mood] ?? 3 });

    // Sentiment trend from AI analysis if present
    const sentiment = j.analysis?.overallSentiment?.toLowerCase();
    if (sentiment) {
      sentimentTrend.push({ date: dt, sentiment, sentimentScore: sentimentMap[sentiment] ?? 3 });
    }

    // Risk trend if present
    const riskScore = j.analysis?.contentAnalysis?.riskScore;
    const risk = j.analysis?.contentAnalysis?.risk?.toLowerCase();
    if (typeof riskScore === 'number') {
      riskTrend.push({ date: dt, riskScore });
    } else if (risk) {
      riskTrend.push({ date: dt, risk: risk, riskScore: riskMap[risk] ?? 1 });
    }

    // Cognitive distortions frequency
    const distortions = j.analysis?.contentAnalysis?.distortions || j.voiceAnalysisResult?.distortions || [];
    for (const d of distortions) {
      const key = String(d).toLowerCase();
      distortionCount[key] = (distortionCount[key] || 0) + 1;
    }

    // Entry frequency
    dateCounts[dt] = (dateCounts[dt] || 0) + 1;
  }

  // Summaries
  const topDistortions = Object.entries(distortionCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([name, count]) => ({ name, count }));

  const entriesOverTime = Object.entries(dateCounts)
    .sort(([d1], [d2]) => (d1 < d2 ? -1 : 1))
    .map(([date, count]) => ({ date, count }));

  // Simple trend direction for mood and sentiment using last 5 points
  const trendDirection = (arr, key) => {
    if (!arr.length) return 'unknown';
    const recent = arr.slice(-5);
    const delta = recent[recent.length - 1][key] - recent[0][key];
    if (delta > 0.5) return 'improving';
    if (delta < -0.5) return 'declining';
    return 'stable';
  };

  const moodDirection = trendDirection(moodsOverTime, 'moodScore');
  const sentimentDirection = trendDirection(sentimentTrend, 'sentimentScore');
  const riskDirection = trendDirection(riskTrend, 'riskScore');

  return {
    count: journals.length,
    dateRange: { start: firstDate, end: lastDate },
    moodsOverTime,
    sentimentTrend,
    riskTrend,
    entriesOverTime,
    topDistortions,
    directions: {
      mood: moodDirection,
      sentiment: sentimentDirection,
      risk: riskDirection,
    },
  };
}
