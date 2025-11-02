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

// Aggregate analytics for cognitive games results
// Input: array of GameResult documents (lean objects)
// Output: summary with per-game series and overall stats
export function computeGameAnalytics(results = []) {
  if (!Array.isArray(results)) return {};

  const byDate = (d) => new Date(d).toISOString().slice(0, 10);
  const perGame = {};

  for (const r of results) {
    const g = r.gameId || 'unknown';
    if (!perGame[g]) {
      perGame[g] = {
        count: 0,
        scores: [],
        scoreSeries: [],
        accuracySeries: [],
        avgReactionSeries: [],
        bestScore: -Infinity,
        avgScore: 0,
        avgAccuracy: 0,
        avgReactionMs: 0,
      };
    }
    const bucket = perGame[g];
    bucket.count += 1;
    const date = byDate(r.createdAt || r.updatedAt || Date.now());
    const score = Number(r.score ?? 0);
    const acc = typeof r.metrics?.accuracy === 'number' ? r.metrics.accuracy : undefined;
    const avgMs = typeof r.metrics?.avgReactionMs === 'number' ? r.metrics.avgReactionMs : undefined;

    bucket.scores.push(score);
    bucket.scoreSeries.push({ date, score });
    if (typeof acc === 'number') bucket.accuracySeries.push({ date, accuracy: acc });
    if (typeof avgMs === 'number') bucket.avgReactionSeries.push({ date, avgReactionMs: avgMs });
    if (score > bucket.bestScore) bucket.bestScore = score;
  }

  // finalize averages
  for (const g of Object.keys(perGame)) {
    const b = perGame[g];
    b.avgScore = b.scores.length ? b.scores.reduce((a, v) => a + v, 0) / b.scores.length : 0;
    const accs = b.accuracySeries.map((x) => x.accuracy);
    b.avgAccuracy = accs.length ? accs.reduce((a, v) => a + v, 0) / accs.length : 0;
    const rts = b.avgReactionSeries.map((x) => x.avgReactionMs);
    b.avgReactionMs = rts.length ? rts.reduce((a, v) => a + v, 0) / rts.length : 0;
  }

  // overall trend across all games
  const overall = {
    totalSessions: results.length,
    byGame: perGame,
  };

  return overall;
}
