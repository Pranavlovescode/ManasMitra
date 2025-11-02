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
    const mood = (j.mood || "neutral").toLowerCase();
    moodsOverTime.push({ date: dt, mood, moodScore: moodMap[mood] ?? 3 });

    // Sentiment trend from AI analysis if present
    const sentiment = j.analysis?.overallSentiment?.toLowerCase();
    if (sentiment) {
      sentimentTrend.push({
        date: dt,
        sentiment,
        sentimentScore: sentimentMap[sentiment] ?? 3,
      });
    }

    // Risk trend if present
    const riskScore = j.analysis?.contentAnalysis?.riskScore;
    const risk = j.analysis?.contentAnalysis?.risk?.toLowerCase();
    if (typeof riskScore === "number") {
      riskTrend.push({ date: dt, riskScore });
    } else if (risk) {
      riskTrend.push({ date: dt, risk: risk, riskScore: riskMap[risk] ?? 1 });
    }

    // Cognitive distortions frequency
    const distortions =
      j.analysis?.contentAnalysis?.distortions ||
      j.voiceAnalysisResult?.distortions ||
      [];
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
    if (!arr.length) return "unknown";
    const recent = arr.slice(-5);
    const delta = recent[recent.length - 1][key] - recent[0][key];
    if (delta > 0.5) return "improving";
    if (delta < -0.5) return "declining";
    return "stable";
  };

  const moodDirection = trendDirection(moodsOverTime, "moodScore");
  const sentimentDirection = trendDirection(sentimentTrend, "sentimentScore");
  const riskDirection = trendDirection(riskTrend, "riskScore");

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
    const g = r.gameId || "unknown";
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
    const acc =
      typeof r.metrics?.accuracy === "number" ? r.metrics.accuracy : undefined;
    const avgMs =
      typeof r.metrics?.avgReactionMs === "number"
        ? r.metrics.avgReactionMs
        : undefined;

    bucket.scores.push(score);
    bucket.scoreSeries.push({ date, score });
    if (typeof acc === "number")
      bucket.accuracySeries.push({ date, accuracy: acc });
    if (typeof avgMs === "number")
      bucket.avgReactionSeries.push({ date, avgReactionMs: avgMs });
    if (score > bucket.bestScore) bucket.bestScore = score;
  }

  // finalize averages
  for (const g of Object.keys(perGame)) {
    const b = perGame[g];
    b.avgScore = b.scores.length
      ? b.scores.reduce((a, v) => a + v, 0) / b.scores.length
      : 0;
    const accs = b.accuracySeries.map((x) => x.accuracy);
    b.avgAccuracy = accs.length
      ? accs.reduce((a, v) => a + v, 0) / accs.length
      : 0;
    const rts = b.avgReactionSeries.map((x) => x.avgReactionMs);
    b.avgReactionMs = rts.length
      ? rts.reduce((a, v) => a + v, 0) / rts.length
      : 0;
  }

  // overall trend across all games
  const overall = {
    totalSessions: results.length,
    byGame: perGame,
  };

  return overall;
}

// Build therapist-facing insights from raw game results.
// Returns cognitive domain scores (0-100), trends, and narrative summary.
export function buildGameInsights(results = []) {
  const analytics = computeGameAnalytics(results);
  const byGame = analytics.byGame || {};

  const GAME_MAP = {
    first: { key: 'first', name: 'Visual Search', domains: ['attention', 'processingSpeed'] },
    second: { key: 'second', name: 'Selective Attention', domains: ['attention'] },
    third_test: { key: 'third_test', name: 'Sequence Memory', domains: ['workingMemory'] },
    fourth: { key: 'fourth', name: 'Verbal Fluency', domains: ['language'] },
    fifth: { key: 'fifth', name: 'Stroop Color Naming', domains: ['executive'] },
    sixth: { key: 'sixth', name: 'Cloze Word Completion', domains: ['language'] },
  };

  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const normalize = (v, min, max, invert = false) => {
    if (typeof v !== 'number' || Number.isNaN(v)) return 0.5;
    if (typeof min !== 'number' || typeof max !== 'number' || min === max) return 0.5;
    const n = clamp01((v - min) / (max - min));
    return invert ? 1 - n : n;
  };

  // Compute per-game last values, ranges, and a unified lastNormalized score per game.
  const perGame = {};
  for (const [gameId, info] of Object.entries(byGame)) {
    const scores = info.scoreSeries || [];
    const accs = info.accuracySeries || [];
    const rts = info.avgReactionSeries || [];

    const lastScore = scores.length ? scores[scores.length - 1].score : undefined;
    const lastAcc = accs.length ? accs[accs.length - 1].accuracy : undefined;
    const lastRt = rts.length ? rts[rts.length - 1].avgReactionMs : undefined;

    const minScore = scores.length ? Math.min(...scores.map((s) => s.score)) : undefined;
    const maxScore = scores.length ? Math.max(...scores.map((s) => s.score)) : undefined;
    const minRt = rts.length ? Math.min(...rts.map((r) => r.avgReactionMs)) : 300; // plausible min
    const maxRt = rts.length ? Math.max(...rts.map((r) => r.avgReactionMs)) : 3000; // plausible max

    // Weighted normalized composite per game: accuracy (0.5), reaction time inverted (0.3), score (0.2)
    const nScore = normalize(lastScore, minScore, maxScore);
    const nAcc = typeof lastAcc === 'number' ? clamp01(lastAcc) : 0.5; // already 0..1
    const nRt = normalize(lastRt, minRt, maxRt, true); // lower is better

    const composite = 0.5 * nAcc + 0.3 * nRt + 0.2 * nScore;

    // Simple trend from last up to 5 points on scoreSeries
    const trendFromSeries = (series, key) => {
      if (!series || !series.length) return 'unknown';
      const recent = series.slice(-5);
      const first = recent[0][key];
      const last = recent[recent.length - 1][key];
      if (typeof first !== 'number' || typeof last !== 'number') return 'unknown';
      const delta = last - first;
      // relative threshold to reduce noise
      const base = Math.max(Math.abs(first), 1);
      const pct = delta / base;
      if (pct > 0.1) return 'improving';
      if (pct < -0.1) return 'declining';
      return 'stable';
    };

    const gameTrend = trendFromSeries(scores, 'score');

    perGame[gameId] = {
      lastScore,
      lastAcc,
      lastRt,
      composite, // 0..1
      trend: gameTrend,
      name: GAME_MAP[gameId]?.name || gameId,
    };
  }

  // Aggregate into cognitive domains
  const domainDefs = {
    attention: {
      label: 'Attention',
      games: ['first', 'second'],
      narrative: 'Focus, selective attention, and filtering distractions',
    },
    executive: {
      label: 'Executive Function',
      games: ['fifth'],
      narrative: 'Inhibitory control and cognitive control under interference',
    },
    workingMemory: {
      label: 'Working Memory',
      games: ['third_test'],
      narrative: 'Holding and manipulating information short-term',
    },
    processingSpeed: {
      label: 'Processing Speed',
      games: ['first'],
      narrative: 'Speed of visual information processing and response',
    },
    language: {
      label: 'Language & Verbal Reasoning',
      games: ['fourth', 'sixth'],
      narrative: 'Word generation, contextual understanding, and verbal fluency',
    },
  };

  const domains = {};
  for (const [key, def] of Object.entries(domainDefs)) {
    const comps = def.games
      .map((g) => perGame[g]?.composite)
      .filter((v) => typeof v === 'number');
    const avg = comps.length ? comps.reduce((a, v) => a + v, 0) / comps.length : 0.5;
    const score = Math.round(avg * 100);

    const trends = def.games.map((g) => perGame[g]?.trend).filter(Boolean);
    const trend = (() => {
      const imp = trends.filter((t) => t === 'improving').length;
      const dec = trends.filter((t) => t === 'declining').length;
      if (imp > dec) return 'improving';
      if (dec > imp) return 'declining';
      return 'stable';
    })();

    const level = score < 40 ? 'low' : score < 70 ? 'average' : 'high';

    domains[key] = {
      key,
      label: def.label,
      narrative: def.narrative,
      score,
      level,
      trend,
    };
  }

  // Narrative summary bullets
  const summary = [];
  const add = (s) => summary.push(s);

  // Highlight strengths and challenges
  const top = Object.values(domains).sort((a, b) => b.score - a.score);
  if (top.length) {
    add(`Strongest domain: ${top[0].label} (${top[0].score}/100, ${top[0].trend}).`);
  }
  const bottom = [...top].reverse();
  if (bottom.length) {
    add(`Greatest challenge: ${bottom[0].label} (${bottom[0].score}/100, ${bottom[0].trend}).`);
  }

  // Condition-related interpretations
  if (domains.executive && domains.executive.level === 'low') {
    add('Low inhibitory control (Stroop) may reflect difficulty suppressing automatic responses, common in heightened anxiety/stress.');
  }
  if (domains.attention && domains.attention.level === 'low') {
    add('Selective/sustained attention appears reduced, which can align with racing thoughts or worry.');
  }
  if (domains.processingSpeed && domains.processingSpeed.level === 'low') {
    add('Processing speed trends are slower; consider fatigue, depressive symptoms, or cognitive load.');
  }
  if (domains.workingMemory && domains.workingMemory.level === 'low') {
    add('Working memory below baseline; this can impact planning and problem-solving in daily tasks.');
  }
  if (domains.language && domains.language.level === 'low') {
    add('Language/fluency challenges may relate to low mood or reduced motivation to engage verbally.');
  }

  // Per-game notes (optional concise notes)
  const perGameNotes = Object.entries(perGame).map(([gid, g]) => {
    const name = GAME_MAP[gid]?.name || gid;
    const trend = g.trend;
    const parts = [];
    if (typeof g.lastAcc === 'number') parts.push(`acc ${Math.round(g.lastAcc * 100)}%`);
    if (typeof g.lastRt === 'number') parts.push(`${Math.round(g.lastRt)} ms`);
    if (typeof g.lastScore === 'number') parts.push(`score ${Math.round(g.lastScore)}`);
    return `${name}: ${trend}, ${parts.join(', ')}`;
  });

  return {
    domains,
    perGame: perGame,
    summary,
    notes: perGameNotes,
  };
}
