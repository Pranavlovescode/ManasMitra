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

    // Find or create patient profile
    let patient = await Patient.findOne({ clerkId: userId });
    
    if (!patient) {
      patient = new Patient({
        userId: user._id,
        clerkId: userId,
        personalInfo: {},
        medicalInfo: {},
        preferences: {},
        consents: {},
        status: { profileComplete: false }
      });
    }

    // Update specific fields from the body
    if (body.personalInfo) {
      Object.assign(patient.personalInfo, body.personalInfo);
    }
    if (body.medicalInfo) {
      Object.assign(patient.medicalInfo, body.medicalInfo);
    }
    if (body.preferences) {
      Object.assign(patient.preferences, body.preferences);
    }
    if (body.guardianInfo) {
      Object.assign(patient.guardianInfo, body.guardianInfo);
    }
    if (body.consents) {
      Object.assign(patient.consents, body.consents);
    }

    // Mark profile as complete
    patient.status.profileComplete = true;
    
    await patient.save();

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