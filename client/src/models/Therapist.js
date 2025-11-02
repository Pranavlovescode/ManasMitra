import mongoose from 'mongoose';

const TherapistSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true,
  },
  clerkId: {
    type: String,
    required: true,
    unique: true,
  },

  // Professional Info
  professionalInfo: {
    licenseNumber: {
      type: String,
      required: true,
    },
    licenseState: String,
    licenseExpiry: Date,
    yearsOfExperience: {
      type: Number,
      min: 0,
      max: 50,
    },
    specializations: [String],
    therapyApproaches: [String],
    languages: [String],
    education: [{
      degree: String,
      institution: String,
      year: Number,
      field: String,
    }],
    certifications: [String],
  },

  // Contact Info
  contactInfo: {
    phoneNumber: String,
    officeAddress: {
      street: String,
      city: String,
      state: String,
      zipCode: String,
      country: { type: String, default: 'US' },
    },
  },

  // Practice Information
  practiceInfo: {
    practiceType: {
      type: String,
      enum: ['private', 'group', 'hospital', 'community', 'other'],
    },
    practiceName: String,
    insurance: [String],
    fees: {
      individual: Number,
      couple: Number,
      group: Number,
      initialConsultation: Number,
    },
  },

  // Personal Information
  personalInfo: {
    bio: String,
    dateOfBirth: Date,
    address: {
      street: String,
      city: String,
      state: String,
      zipCode: String,
      country: { type: String, default: 'US' },
    },
    emergencyContact: {
      name: String,
      phone: String,
      relationship: String,
    },
    profilePhoto: String,
  },

  // Preferences
  preferences: {
    sessionFormats: [{
      type: String,
      enum: ['in-person', 'virtual', 'both'],
    }],
    acceptingNewPatients: {
      type: Boolean,
      default: true,
    },
    availabilityHours: [{
      day: String,
      startTime: String,
      endTime: String,
    }],
  },

  // Status
  status: {
    profileComplete: { type: Boolean, default: false },
    isActive: { type: Boolean, default: true },
    verified: { type: Boolean, default: false },
  },
}, {
  timestamps: true,
  strict: false, // Allow additional fields for flexibility
});

// Indexes
TherapistSchema.index({ 'status.isActive': 1 });
TherapistSchema.index({ 'status.verified': 1 });
TherapistSchema.index({ 'preferences.acceptingNewPatients': 1 });

// Virtual for full name
TherapistSchema.virtual('fullName').get(function () {
  const user = this.populated('userId') || this.userId;
  if (user?.firstName && user?.lastName) return `${user.firstName} ${user.lastName}`;
  return 'Unknown Therapist';
});

// Ensure virtual fields are serialized
TherapistSchema.set('toJSON', { virtuals: true });

export default mongoose.models.Therapist || mongoose.model('Therapist', TherapistSchema);