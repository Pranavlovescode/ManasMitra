import mongoose from 'mongoose';

// ========================
// Patient Schema (Simplified)
// ========================
const PatientSchema = new mongoose.Schema({
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

  // Personal Info
  personalInfo: {
    dateOfBirth: Date,
    gender: {
      type: String,
      enum: ['male', 'female', 'other', 'prefer-not-to-say'],
      default: 'prefer-not-to-say',
    },
    phoneNumber: String,
    address: {
      street: String,
      city: String,
      state: String,
      zipCode: String,
      country: { type: String, default: 'India' },
    },
    occupation: String,
    maritalStatus: {
      type: String,
      enum: ['single', 'married', 'divorced', 'widowed', 'separated', 'other'],
      default: 'single',
    },
  },

  // Guardian Info (Optional)
  guardianInfo: {
    isMinor: { type: Boolean, default: false },
    guardians: [{
      firstName: String,
      lastName: String,
      relationship: String,
      phoneNumber: String,
      email: String,
      address: {
        street: String,
        city: String,
        state: String,
        zipCode: String,
        country: { type: String, default: 'India' },
      },
      isPrimary: { type: Boolean, default: false },
    }],
  },

  // Medical Info (Optional)
  medicalInfo: {
    primaryPhysician: {
      name: String,
      phone: String,
      email: String,
    },
    allergies: [String],
    currentMedications: [{
      name: String,
      dosage: String,
      frequency: String,
      prescribedBy: String,
    }],
    medicalConditions: [String],
    previousTherapy: {
      hasHadTherapy: { type: Boolean, default: false },
      therapistName: String,
      duration: String,
      reason: String,
    },
  },

  // Insurance Info (Optional)
  insuranceInfo: {
    hasInsurance: { type: Boolean, default: false },
    provider: String,
    policyNumber: String,
    groupNumber: String,
    subscriberName: String,
    subscriberRelationship: String,
  },

  // Preferences
  preferences: {
    preferredContactMethod: {
      type: String,
      enum: ['email', 'phone', 'text'],
      default: 'email',
    },
    therapistGenderPreference: {
      type: String,
      enum: ['male', 'female', 'no-preference'],
      default: 'no-preference',
    },
    sessionFormat: {
      type: String,
      enum: ['in-person', 'virtual', 'both'],
      default: 'both',
    },
    availabilityTimes: [{
      day: String,
      startTime: String,
      endTime: String,
    }],
  },

  // Consents
  consents: {
    treatmentConsent: { type: Boolean, default: false },
    hipaaConsent: { type: Boolean, default: false },
    communicationConsent: { type: Boolean, default: false },
    guardianConsent: { type: Boolean, default: false },
    consentDate: Date,
  },

  // Status
  status: {
    profileComplete: { type: Boolean, default: false },
    activePatient: { type: Boolean, default: true },
    assignedTherapist: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  },
}, {
  timestamps: true,
});

// ========================
// Virtuals
// ========================
PatientSchema.virtual('age').get(function () {
  if (!this.personalInfo?.dateOfBirth) return null;
  const today = new Date();
  const dob = new Date(this.personalInfo.dateOfBirth);
  let age = today.getFullYear() - dob.getFullYear();
  const m = today.getMonth() - dob.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) age--;
  return age;
});

PatientSchema.virtual('fullName').get(function () {
  const user = this.populated('userId') || this.userId;
  if (user?.firstName && user?.lastName) return `${user.firstName} ${user.lastName}`;
  return 'Unknown Patient';
});

// ========================
// Methods
// ========================
PatientSchema.methods.isMinor = function () {
  return this.age !== null && this.age < 18;
};

// ========================
// Indexes
// ========================
PatientSchema.index({ 'status.activePatient': 1 });
PatientSchema.index({ 'status.assignedTherapist': 1 });

// ========================
// Export
// ========================
delete mongoose.models.Patient; // Prevent model overwrite in dev
export default mongoose.model('Patient', PatientSchema);
