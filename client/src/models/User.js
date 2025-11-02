import mongoose from 'mongoose';

const UserSchema = new mongoose.Schema({
  clerkId: {
    type: String,
    required: true,
    unique: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  firstName: {
    type: String,
    required: true,
    trim: true
  },
  lastName: {
    type: String,
    required: true,
    trim: true
  },
  role: {
    type: String,
    enum: ['patient', 'therapist'],
    required: true
  },
  profileComplete: {
    type: Boolean,
    default: false
  },
  profileImage: String,
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true,
});

// Static method to create a safe user object (only allowed fields)
UserSchema.statics.createSafeUser = function(userData) {
  const allowedFields = ['clerkId', 'email', 'firstName', 'lastName', 'role', 'profileImage', 'profileComplete', 'isActive'];
  const safeData = {};
  
  // Only include allowed fields from the schema
  for (const field of allowedFields) {
    if (userData[field] !== undefined) {
      safeData[field] = userData[field];
    }
  }
  
  // Ensure required fields have values
  if (!safeData.firstName) safeData.firstName = '';
  if (!safeData.lastName) safeData.lastName = '';
  if (!safeData.role) safeData.role = 'patient';
  
  return new this(safeData);
};

// Static method to safely update user with only allowed fields
UserSchema.statics.updateSafeUser = function(userDoc, updateData) {
  const allowedFields = ['email', 'firstName', 'lastName', 'profileImage', 'profileComplete'];
  
  // Only update allowed fields
  for (const field of allowedFields) {
    if (updateData[field] !== undefined) {
      userDoc[field] = updateData[field];
    }
  }
  
  return userDoc;
};

// Static method to return clean user data (remove sensitive fields)
UserSchema.statics.getCleanUserData = function(userDoc) {
  const userObj = userDoc.toObject ? userDoc.toObject() : userDoc;
  
  // Remove sensitive or internal fields
  const cleanUser = {
    _id: userObj._id,
    clerkId: userObj.clerkId,
    email: userObj.email,
    firstName: userObj.firstName,
    lastName: userObj.lastName,
    role: userObj.role,
    profileComplete: userObj.profileComplete,
    profileImage: userObj.profileImage,
    isActive: userObj.isActive !== false, // Default to true if undefined
    createdAt: userObj.createdAt,
    updatedAt: userObj.updatedAt
  };
  
  return cleanUser;
};

// Instance method to get clean data
UserSchema.methods.getCleanData = function() {
  return this.constructor.getCleanUserData(this);
};

export default mongoose.models.User || mongoose.model('User', UserSchema);
