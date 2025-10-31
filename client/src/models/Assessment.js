import mongoose from 'mongoose';

const assessmentSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
  },
  date: {
    type: Date,
    default: Date.now,
  },
  // Original assessments (backward compatibility)
  gad7: {
    score: Number,
    severity: String,
    answers: [{
      questionId: Number,
      answer: Number
    }]
  },
  phq9: {
    score: Number,
    severity: String,
    answers: [{
      questionId: Number,
      answer: Number
    }]
  },
  // New assessment types
  stress: {
    score: Number,
    severity: String,
    answers: [{
      questionId: Number,
      answer: Number
    }]
  },
  sleep: {
    score: Number,
    severity: String,
    answers: [{
      questionId: Number,
      answer: Number
    }]
  }
});

export default mongoose.models.Assessment || mongoose.model('Assessment', assessmentSchema);