import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';

// GET /api/patients/therapist - Get the therapist assigned to current patient
export async function GET() {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();
    
    // Find the patient
    const patient = await Patient.findOne({ clerkId: userId })
      .populate('medicalInfo.assignedTherapist', 'firstName lastName email');
    
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const assignedTherapist = patient.medicalInfo?.assignedTherapist;

    return NextResponse.json({ 
      hasAssignedTherapist: !!assignedTherapist,
      therapist: assignedTherapist ? {
        _id: assignedTherapist._id,
        name: `${assignedTherapist.firstName} ${assignedTherapist.lastName}`,
        email: assignedTherapist.email
      } : null
    });

  } catch (error) {
    console.error('Error fetching patient therapist:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// PUT /api/patients/therapist - Update the therapist assigned to current patient  
export async function PUT(request) {
  try {
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { therapistId } = await request.json();

    await connectDB();
    
    // Validate therapist exists and is active
    if (therapistId) {
      const therapist = await User.findOne({ 
        _id: therapistId, 
        role: 'therapist',
        isActive: true 
      });
      
      if (!therapist) {
        return NextResponse.json({ error: 'Invalid therapist selection' }, { status: 400 });
      }
    }

    // Update patient's assigned therapist
    const patient = await Patient.findOneAndUpdate(
      { clerkId: userId },
      { 'medicalInfo.assignedTherapist': therapistId || null },
      { new: true }
    ).populate('medicalInfo.assignedTherapist', 'firstName lastName email');
    
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const assignedTherapist = patient.medicalInfo?.assignedTherapist;

    return NextResponse.json({ 
      message: 'Therapist assignment updated successfully',
      hasAssignedTherapist: !!assignedTherapist,
      therapist: assignedTherapist ? {
        _id: assignedTherapist._id,
        name: `${assignedTherapist.firstName} ${assignedTherapist.lastName}`,
        email: assignedTherapist.email
      } : null
    });

  } catch (error) {
    console.error('Error updating patient therapist:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}