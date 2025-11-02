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
    
    const body = await req.json();
    console.log('📋 Profile completion request:', JSON.stringify(body, null, 2));
    
    await connectDB();
    
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Mark profile as complete
    user.profileComplete = true;
    
    // Update any additional allowed fields if provided
    User.updateSafeUser(user, body);
    
    await user.save();
    
    console.log('✅ Profile completed for user:', user.email);

    return NextResponse.json({ 
      message: 'Profile completed successfully', 
      user: User.getCleanUserData(user)
    });

  } catch (error) {
    console.error('Error completing profile:', error);
    
    // Check if it's a validation error due to extra fields
    if (error.name === 'StrictModeError') {
      return NextResponse.json({ 
        error: 'Invalid data provided. Only allowed fields can be updated.',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      }, { status: 400 });
    }
    
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
      user: User.getCleanUserData(user)
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