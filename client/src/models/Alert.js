import mongoose from 'mongoose';

const AlertSchema = new mongoose.Schema({
  therapistId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, index: true },
  patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true, index: true },
  patientClerkId: { type: String, required: true, index: true },
  patientName: { type: String },
  title: { type: String, required: true },
  message: { type: String, required: true },
  severity: { type: String, enum: ['low', 'medium', 'high'], default: 'medium' },
  meta: { type: mongoose.Schema.Types.Mixed },
  dismissed: { type: Boolean, default: false },
}, { timestamps: true });

AlertSchema.index({ therapistId: 1, dismissed: 1, createdAt: -1 });

export default mongoose.models.Alert || mongoose.model('Alert', AlertSchema);
