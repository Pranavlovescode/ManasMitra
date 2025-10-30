import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';

// GET /api/patients/profile - Get current patient profile
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

    // Check if user is a patient
    if (user.role !== 'patient') {
      return NextResponse.json({ error: 'Access denied. Only patients can access this endpoint.' }, { status: 403 });
    }

    // Find patient details
    const patient = await Patient.findOne({ clerkId: userId }).populate('userId');
    
    return NextResponse.json({
      hasDetails: !!patient,
      patient: patient || null,
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        profileComplete: user.profileComplete
      }
    });
  } catch (error) {
    console.error('Error fetching patient profile:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// PUT /api/patients/profile - Update patient profile
export async function PUT(req) {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    
    await connectDB();
    
    // Find the user first
    const user = await User.findOne({ clerkId: userId });
    
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // Check if user is a patient
    if (user.role !== 'patient') {
      return NextResponse.json({ error: 'Access denied. Only patients can update patient details.' }, { status: 403 });
    }

    // Find and update patient details
    let patient = await Patient.findOneAndUpdate(
      { clerkId: userId },
      { ...body },
      { new: true, upsert: true }
    );

    // Update user profile completion status
    await User.findByIdAndUpdate(user._id, { profileComplete: true });

    // Populate the userId field before returning
    await patient.populate('userId');

    return NextResponse.json({
      message: 'Patient profile updated successfully',
      patient
    });
  } catch (error) {
    console.error('Error updating patient profile:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}