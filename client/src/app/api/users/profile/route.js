import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';

// POST /api/users/profile - Complete user profile
export async function POST(req) {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const profileData = await req.json();
    
    await connectDB();
    
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Validate required fields based on role
    if (user.role === 'patient') {
      const { dateOfBirth, emergencyContact } = profileData;
      
      if (!dateOfBirth) {
        return NextResponse.json({ error: 'Date of birth is required for patients' }, { status: 400 });
      }
      
      user.dateOfBirth = new Date(dateOfBirth);
      
      if (emergencyContact) {
        user.emergencyContact = emergencyContact;
      }
      
      if (profileData.medicalHistory) {
        user.medicalHistory = profileData.medicalHistory;
      }
      
    } else if (user.role === 'therapist') {
      const { licenseNumber, specializations, yearsOfExperience, education } = profileData;
      
      if (!licenseNumber || !yearsOfExperience) {
        return NextResponse.json({ 
          error: 'License number and years of experience are required for therapists' 
        }, { status: 400 });
      }
      
      user.licenseNumber = licenseNumber;
      user.yearsOfExperience = yearsOfExperience;
      
      if (specializations) {
        user.specializations = specializations;
      }
      
      if (education) {
        user.education = education;
      }
      
      if (profileData.acceptingPatients !== undefined) {
        user.acceptingPatients = profileData.acceptingPatients;
      }
    }

    // Update common fields
    if (profileData.phone) user.phone = profileData.phone;
    if (profileData.address) user.address = profileData.address;
    if (profileData.preferences) user.preferences = { ...user.preferences, ...profileData.preferences };

    // Mark profile as complete
    user.profileComplete = true;
    user.lastLogin = new Date();

    await user.save();

    return NextResponse.json({ 
      message: 'Profile completed successfully', 
      user 
    });

  } catch (error) {
    console.error('Error completing profile:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// GET /api/users/profile - Check if profile is complete
export async function GET() {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    return NextResponse.json({ 
      profileComplete: user.profileComplete,
      role: user.role,
      requiredFields: getRequiredFields(user.role),
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        profileComplete: user.profileComplete
      }
    });

  } catch (error) {
    console.error('Error checking profile:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

function getRequiredFields(role) {
  if (role === 'patient') {
    return {
      dateOfBirth: 'Date of Birth',
      emergencyContact: {
        name: 'Emergency Contact Name',
        phone: 'Emergency Contact Phone',
        relationship: 'Relationship'
      }
    };
  } else if (role === 'therapist') {
    return {
      licenseNumber: 'License Number',
      yearsOfExperience: 'Years of Experience',
      specializations: 'Specializations (optional)',
      education: 'Education (optional)'
    };
  }
  return {};
}