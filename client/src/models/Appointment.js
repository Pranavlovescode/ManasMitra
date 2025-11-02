import mongoose from 'mongoose';

const AppointmentSchema = new mongoose.Schema({
  // Patient info
  patientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Patient',
    required: true,
  },
  patientClerkId: {
    type: String,
    required: true,
  },

  // Therapist info
  therapistId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Therapist',
    required: true,
  },
  therapistClerkId: {
    type: String,
    required: true,
  },

  // Appointment details
  scheduledAt: {
    type: Date,
    required: true,
  },
  duration: {
    type: Number, // in minutes
    default: 60,
  },
  type: {
    type: String,
    enum: ['individual', 'group', 'couple', 'family'],
    default: 'individual',
  },

  // Status tracking
  status: {
    type: String,
    enum: ['pending', 'accepted', 'declined', 'completed', 'cancelled', 'rescheduled'],
    default: 'pending',
  },

  // Additional details
  notes: String,
  patientNotes: String, // Notes from patient when booking
  therapistNotes: String, // Notes from therapist

  // Meeting details
  meetingLink: String,
  meetingType: {
    type: String,
    enum: ['virtual', 'in-person'],
    default: 'virtual',
  },

  // Modification tracking
  originalScheduledAt: Date, // Store original time if rescheduled
  modifiedBy: {
    type: String,
    enum: ['patient', 'therapist', 'system'],
  },
  modifiedAt: Date,

  // Cancellation
  cancelledBy: {
    type: String,
    enum: ['patient', 'therapist'],
  },
  cancellationReason: String,
  cancelledAt: Date,
}, {
  timestamps: true,
});

// Indexes for efficient queries
AppointmentSchema.index({ patientClerkId: 1, status: 1, scheduledAt: -1 });
AppointmentSchema.index({ therapistClerkId: 1, status: 1, scheduledAt: -1 });
AppointmentSchema.index({ scheduledAt: 1 });
AppointmentSchema.index({ status: 1 });

// Virtual for checking if appointment is upcoming
AppointmentSchema.virtual('isUpcoming').get(function() {
  return this.scheduledAt > new Date() && this.status === 'accepted';
});

// Virtual for checking if appointment is past
AppointmentSchema.virtual('isPast').get(function() {
  return this.scheduledAt < new Date();
});

AppointmentSchema.set('toJSON', { virtuals: true });
AppointmentSchema.set('toObject', { virtuals: true });

export default mongoose.models.Appointment || mongoose.model('Appointment', AppointmentSchema);
