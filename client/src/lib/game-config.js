/**
 * Centralized Game Configuration
 * Maps game IDs to proper names, descriptions, and cognitive domains
 */

export const GAMES = [
  {
    id: "first",
    title: "Visual Search",
    desc: "Find target shapes under time pressure.",
    emoji: "🎯",
    cognitiveDomain: "Attention",
    skills: ["Visual Processing", "Selective Attention", "Processing Speed"],
  },
  {
    id: "second",
    title: "Selective Attention",
    desc: "Track moving targets and count accurately.",
    emoji: "👀",
    cognitiveDomain: "Attention",
    skills: ["Sustained Attention", "Divided Attention", "Focus"],
  },
  {
    id: "third_test",
    title: "Sequence Memory",
    desc: "Memorize and recall sequences.",
    emoji: "🧠",
    cognitiveDomain: "Memory",
    skills: ["Working Memory", "Sequential Processing", "Recall"],
  },
  {
    id: "fourth",
    title: "Verbal Fluency",
    desc: "Generate valid words fast.",
    emoji: "🔤",
    cognitiveDomain: "Language",
    skills: ["Verbal Fluency", "Word Retrieval", "Language Processing"],
  },
  {
    id: "fifth",
    title: "Stroop Color Naming",
    desc: "Pick the color, not the word.",
    emoji: "🎨",
    cognitiveDomain: "Executive Function",
    skills: ["Inhibitory Control", "Cognitive Flexibility", "Conflict Resolution"],
  },
  {
    id: "sixth",
    title: "Cloze Word Completion",
    desc: "Fill in missing letters to form words.",
    emoji: "✍️",
    cognitiveDomain: "Language",
    skills: ["Word Recognition", "Pattern Completion", "Semantic Processing"],
  },
];

// Quick lookup map: ID -> Game object
export const GAME_MAP = GAMES.reduce((acc, game) => {
  acc[game.id] = game;
  return acc;
}, {});

// Game ID to name mapping (for backward compatibility)
export const GAME_NAMES = {
  first: "Visual Search",
  second: "Selective Attention",
  third_test: "Sequence Memory",
  fourth: "Verbal Fluency",
  fifth: "Stroop Color Naming",
  sixth: "Cloze Word Completion",
};

// Cognitive domain colors for visualization
export const DOMAIN_COLORS = {
  Attention: {
    bg: "bg-blue-50",
    border: "border-blue-500",
    text: "text-blue-700",
    chart: "rgba(59, 130, 246, 0.8)",
  },
  Memory: {
    bg: "bg-purple-50",
    border: "border-purple-500",
    text: "text-purple-700",
    chart: "rgba(147, 51, 234, 0.8)",
  },
  Language: {
    bg: "bg-green-50",
    border: "border-green-500",
    text: "text-green-700",
    chart: "rgba(34, 197, 94, 0.8)",
  },
  "Executive Function": {
    bg: "bg-orange-50",
    border: "border-orange-500",
    text: "text-orange-700",
    chart: "rgba(249, 115, 22, 0.8)",
  },
};

/**
 * Get game information by ID
 * @param {string} gameId - The game ID (e.g., "first", "second")
 * @returns {object|null} Game object or null if not found
 */
export function getGameById(gameId) {
  return GAME_MAP[gameId] || null;
}

/**
 * Get game name by ID
 * @param {string} gameId - The game ID
 * @returns {string} Game name or the original ID if not found
 */
export function getGameName(gameId) {
  return GAME_NAMES[gameId] || gameId;
}

/**
 * Get cognitive domain for a game
 * @param {string} gameId - The game ID
 * @returns {string} Cognitive domain name
 */
export function getCognitiveDomain(gameId) {
  const game = GAME_MAP[gameId];
  return game ? game.cognitiveDomain : "Unknown";
}

/**
 * Group games by cognitive domain
 * @returns {object} Games grouped by domain
 */
export function getGamesByDomain() {
  return GAMES.reduce((acc, game) => {
    const domain = game.cognitiveDomain;
    if (!acc[domain]) {
      acc[domain] = [];
    }
    acc[domain].push(game);
    return acc;
  }, {});
}

/**
 * Get domain color configuration
 * @param {string} domain - Cognitive domain name
 * @returns {object} Color configuration object
 */
export function getDomainColors(domain) {
  return DOMAIN_COLORS[domain] || {
    bg: "bg-gray-50",
    border: "border-gray-500",
    text: "text-gray-700",
    chart: "rgba(107, 114, 128, 0.8)",
  };
}
