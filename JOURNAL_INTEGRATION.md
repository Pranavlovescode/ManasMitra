# Journal Analysis Integration

This document describes the integration between the Next.js journal API and the FastAPI CBT analysis backend.

## Architecture Overview

```
Frontend (Next.js) → API Route (/api/journal) → FastAPI (/analyze/journal) → CBT Models
                                    ↓
                            MongoDB (Journal + Analysis)
```

## Components

### 1. FastAPI Backend (`main.py`)

**New Endpoint**: `POST /analyze/journal`

**Request Format**:
```json
{
  "title": "Journal entry title",
  "content": "Journal entry content...",
  "mood": "sad|neutral|happy|excited|loved",
  "prompt": "Optional writing prompt"
}
```

**Response Format**:
```json
{
  "content_analysis": {
    "emotion": "anxious",
    "emotion_score": 0.85,
    "intent": "seeking_help",
    "intent_score": 0.72,
    "risk": "moderate",
    "risk_score": 0.35,
    "distortions": ["catastrophizing", "all_or_nothing"],
    "distortion_details": [...],
    "reframes": [...],
    "behavioral_suggestions": [...],
    "clinician_notes": [...]
  },
  "title_analysis": {...},
  "overall_sentiment": "negative",
  "key_themes": ["Anxiety", "Work/Career"],
  "therapeutic_insights": [...],
  "progress_indicators": [...],
  "recommendations": [...]
}
```

### 2. Next.js API Route (`/api/journal/route.js`)

**Enhanced Features**:
- Automatically analyzes journal entries on creation
- Stores analysis results in MongoDB
- Graceful fallback if FastAPI is unavailable
- Optional `skipAnalysis` parameter for performance

**New Parameters**:
- `skipAnalysis`: boolean - Skip CBT analysis
- `reanalyze`: boolean - Force re-analysis on updates

### 3. MongoDB Journal Model

**New Fields Added**:
```javascript
analysis: {
  contentAnalysis: {
    emotion: String,
    emotionScore: Number,
    intent: String,
    // ... full analysis results
  },
  overallSentiment: String,
  keyThemes: [String],
  therapeuticInsights: [String],
  progressIndicators: [String],
  recommendations: [String],
  analysisTimestamp: Date
}
```

### 4. Additional API Route (`/api/journal/analyze/route.js`)

**Purpose**: Analyze existing journal entries
**Endpoint**: `POST /api/journal/analyze`
**Request**: `{ "journalId": "mongoId" }`

## Usage Examples

### 1. Create Journal with Analysis
```javascript
const response = await fetch('/api/journal', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "My day",
    content: "I feel overwhelmed with everything...",
    mood: "sad",
    selectedPrompt: "How was your day?"
  })
});
```

### 2. Create Journal without Analysis
```javascript
const response = await fetch('/api/journal', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "Quick note",
    content: "Just a quick thought",
    skipAnalysis: true
  })
});
```

### 3. Analyze Existing Journal
```javascript
const response = await fetch('/api/journal/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    journalId: "65f7a8b9c8d2e1f3a4b5c6d7"
  })
});
```

### 4. Update Journal with Re-analysis
```javascript
const response = await fetch('/api/journal', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    journalId: "65f7a8b9c8d2e1f3a4b5c6d7",
    content: "Updated content...",
    reanalyze: true
  })
});
```

## Testing

1. **Start FastAPI Server**:
   ```bash
   python main.py
   ```

2. **Test Integration**:
   ```bash
   node test-journal-integration.js
   ```

3. **Check Health**:
   - Visit: http://127.0.0.1:8000/health
   - Visit: http://127.0.0.1:8000/models/status

## Error Handling

- **FastAPI Unavailable**: Journals are saved without analysis
- **Analysis Failure**: Error is logged but journal creation continues
- **Invalid Input**: Proper validation and error responses
- **Authentication**: All routes require valid Clerk authentication

## Performance Considerations

- Analysis adds ~1-3 seconds to journal creation
- Use `skipAnalysis: true` for bulk operations
- Consider background processing for large volumes
- Analysis results are cached in MongoDB

## Security

- All routes require Clerk authentication
- Users can only access their own journals
- FastAPI runs locally (127.0.0.1:8000)
- No sensitive data in analysis requests

## Monitoring

- Check FastAPI logs for analysis errors
- Monitor Next.js API route performance
- Track analysis success/failure rates
- Monitor MongoDB storage growth

## Future Enhancements

- Background analysis processing
- Bulk analysis for historical journals
- Analysis result comparison over time
- Enhanced therapeutic insights
- Integration with therapist dashboard