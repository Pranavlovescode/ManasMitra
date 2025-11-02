import mongoose from 'mongoose';

const GameResultSchema = new mongoose.Schema(
  {
    userId: { type: String, required: true, index: true }, // Clerk ID
    gameId: {
      type: String,
      required: true,
      enum: ['first', 'second', 'third_test', 'fourth', 'fifth', 'sixth'],
      index: true,
    },
    score: { type: Number, default: 0 },
    // Generic metrics bag so different games can persist different fields
    metrics: {
      rightClicks: Number,
      wrongClicks: Number,
      accuracy: Number, // 0..1
      avgReactionMs: Number,
      reactionTimes: [Number],
      rounds: Number,
      totalCorrect: Number,
      totalWrong: Number,
      totalAttempts: Number,
    },
  },
  { timestamps: true }
);

export default mongoose.models.GameResult || mongoose.model('GameResult', GameResultSchema);
