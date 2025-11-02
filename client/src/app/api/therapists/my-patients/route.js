import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';

// GET /api/therapists/patients - Get all patients assigned to the current therapist
export async function GET(req) {
  try {
    const { userId } = await auth(req);
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Verify the user is a therapist
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    // Find all patients assigned to this therapist
    const patients = await Patient.find({ 
      'medicalInfo.assignedTherapist': therapist._id,
      'status.activePatient': true
    }).populate('userId', 'firstName lastName email');

    const patientList = patients.map(patient => ({
      _id: patient._id,
      name: `${patient.userId?.firstName || ''} ${patient.userId?.lastName || ''}`.trim(),
      email: patient.userId?.email,
      dateOfBirth: patient.personalInfo?.dateOfBirth,
      phoneNumber: patient.personalInfo?.phoneNumber,
      profileComplete: patient.status?.profileComplete,
      createdAt: patient.createdAt,
      preferences: {
        sessionFormat: patient.preferences?.sessionFormat,
        preferredContactMethod: patient.preferences?.preferredContactMethod
      }
    }));

    return NextResponse.json({ 
      patients: patientList,
      totalPatients: patientList.length 
    });

  } catch (error) {
    console.error('Error fetching therapist patients:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}