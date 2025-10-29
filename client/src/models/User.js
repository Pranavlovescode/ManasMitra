import mongoose from 'mongoose';

const UserSchema = new mongoose.Schema({
  clerkId: {
    type: String,
    required: true,
    unique: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  firstName: {
    type: String,
    required: true,
  },
  lastName: {
    type: String,
    required: true,
  },
  role: {
    type: String,
    enum: ['patient', 'therapist'],
    required: true,
  },
  profileComplete: {
    type: Boolean,
    default: false,
  },
  // Patient-specific fields
  dateOfBirth: {
    type: Date,
    required: false
  },
  emergencyContact: {
    name: String,
    phone: String,
    relationship: String,
  },
  medicalHistory: [{
    condition: String,
    diagnosis: Date,
    medications: [String],
  }],
  
  // Therapist-specific fields
  licenseNumber: {
    type: String,
    required: false
  },
  specializations: [String],
  yearsOfExperience: {
    type: Number,
    required: false
  },
  education: [{
    degree: String,
    institution: String,
    year: Number,
  }],
  acceptingPatients: {
    type: Boolean,
    default: true,
  },
  
  // Common fields
  avatar: String,
  profileImage: String, // For Clerk profile image URL
  phone: String,
  address: {
    street: String,
    city: String,
    state: String,
    zipCode: String,
    country: {
      type: String,
      default: 'US'
    }
  },
  preferences: {
    notifications: {
      email: { type: Boolean, default: true },
      sms: { type: Boolean, default: false },
      push: { type: Boolean, default: true },
    },
    theme: {
      type: String,
      enum: ['light', 'dark', 'auto'],
      default: 'auto'
    },
    language: {
      type: String,
      default: 'en'
    }
  },
  
  // Timestamps
  lastLogin: Date,
  isActive: {
    type: Boolean,
    default: true,
  },
}, {
  timestamps: true, // This adds createdAt and updatedAt automatically
});

// Create indexes for better performance (excluding duplicates)
// clerkId and email indexes are already created by unique: true
UserSchema.index({ role: 1 });
UserSchema.index({ 'licenseNumber': 1 }, { sparse: true });

// Virtual for full name
UserSchema.virtual('fullName').get(function() {
  return `${this.firstName} ${this.lastName}`;
});

// Ensure virtual fields are serialized
UserSchema.set('toJSON', { virtuals: true });

export default mongoose.models.User || mongoose.model('User', UserSchema);