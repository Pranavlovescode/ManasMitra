import { auth, getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Therapist from '@/models/Therapist';

// GET /api/therapists/profile - Get current therapist profile
export async function GET() {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Find the user first
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Check if user is a therapist
    if (user.role !== 'therapist') {
      return NextResponse.json({ error: 'Access denied. Only therapists can access this endpoint.' }, { status: 403 });
    }

    // Find therapist details
    const therapist = await Therapist.findOne({ clerkId: userId }).populate('userId');
    
    return NextResponse.json({
      hasDetails: !!therapist,
      therapist: therapist || null,
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        profileComplete: user.profileComplete
      }
    });
  } catch (error) {
    console.error('Error fetching therapist profile:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// PUT /api/therapists/profile - Update therapist profile
export async function PUT(req) {
  try {
    const { userId } = getAuth(req);
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    console.log('📥 Therapist profile data received:', body);
    
    await connectDB();
    
    // Find the user first
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Check if user is a therapist
    if (user.role !== 'therapist') {
      return NextResponse.json({ error: 'Access denied. Only therapists can update therapist details.' }, { status: 403 });
    }

    // Find or create therapist profile
    let therapist = await Therapist.findOne({ clerkId: userId });
    
    if (!therapist) {
      therapist = new Therapist({
        userId: user._id,
        clerkId: userId,
        professionalInfo: {},
        contactInfo: {},
        preferences: {},
        status: { profileComplete: false }
      });
    }

    // Update specific fields from the body
    if (body.professionalInfo) {
      Object.assign(therapist.professionalInfo, body.professionalInfo);
    }
    if (body.contactInfo) {
      Object.assign(therapist.contactInfo, body.contactInfo);
    }
    if (body.preferences) {
      Object.assign(therapist.preferences, body.preferences);
    }

    // Update additional fields if provided
    if (body.practiceInfo) {
      therapist.practiceInfo = body.practiceInfo;
    }
    if (body.personalInfo) {
      therapist.personalInfo = body.personalInfo;
    }

    // Update User model with basic info
    if (body.firstName) user.firstName = body.firstName;
    if (body.lastName) user.lastName = body.lastName;
    if (body.personalInfo?.dateOfBirth) user.dateOfBirth = body.personalInfo.dateOfBirth;
    if (body.personalInfo?.address) user.address = body.personalInfo.address;
    if (body.contactInfo?.phoneNumber) user.phone = body.contactInfo.phoneNumber;

    // Mark profile as complete
    therapist.status.profileComplete = true;
    user.profileComplete = true;
    
    await therapist.save();
    await user.save();

    // Populate the userId field before returning
    await therapist.populate('userId');

    console.log('✅ Therapist profile saved successfully');

    return NextResponse.json({
      message: 'Therapist profile updated successfully',
      therapist,
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        profileComplete: user.profileComplete
      }
    });
  } catch (error) {
    console.error('❌ Error updating therapist profile:', error);
    return NextResponse.json({ 
      error: 'Failed to update therapist profile',
      details: error.message 
    }, { status: 500 });
  }
}