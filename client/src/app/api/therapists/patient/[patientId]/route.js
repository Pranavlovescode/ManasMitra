import { getAuth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '../../../../../lib/mongodb';
import Patient from '../../../../../models/Patient';

export async function GET(request, { params }) {
  try {
    const auth = getAuth(request);
    const { userId } = auth;

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { patientId } = params;

    // Get patient details
    const patient = await Patient.findById(patientId)
      .populate('userId', 'firstName lastName email profileImage');

    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    return NextResponse.json({
      _id: patient._id,
      userId: patient.userId,
      personalInfo: patient.personalInfo,
      medicalInfo: patient.medicalInfo,
      preferences: patient.preferences,
      guardianInfo: patient.guardianInfo,
      insuranceInfo: patient.insuranceInfo,
      status: patient.status,
      age: patient.age,
      createdAt: patient.createdAt,
      updatedAt: patient.updatedAt,
    });
  } catch (error) {
    console.error('Error fetching patient details:', error);
    return NextResponse.json({ 
      error: 'Internal Server Error',
      message: error.message 
    }, { status: 500 });
  }
}
