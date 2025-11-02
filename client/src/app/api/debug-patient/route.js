import { auth } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';

// GET /api/debug-patient - Debug endpoint to check patient data
export async function GET(req) {
  try {
    const { userId } = await auth(req);
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

  await connectDB();
    
    // Find user in Users collection
    const user = await User.findOne({ clerkId: userId });
    
    // Find patient in Patient collection
    const patient = await Patient.findOne({ clerkId: userId })
      .populate('medicalInfo.assignedTherapist', 'firstName lastName email');
    
    // Also try to find patient by userId reference
    const patientByUserId = user ? await Patient.findOne({ userId: user._id })
      .populate('medicalInfo.assignedTherapist', 'firstName lastName email') : null;
    
    return NextResponse.json({
      debug: {
        clerkUserId: userId,
        userFound: !!user,
        user: user ? {
          _id: user._id,
          clerkId: user.clerkId,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role
        } : null,
        patientFoundByClerkId: !!patient,
        patientByClerkId: patient ? {
          _id: patient._id,
          clerkId: patient.clerkId,
          userId: patient.userId,
          hasAssignedTherapist: !!patient.medicalInfo?.assignedTherapist,
          assignedTherapist: patient.medicalInfo?.assignedTherapist
        } : null,
        patientFoundByUserId: !!patientByUserId,
        patientByUserId: patientByUserId ? {
          _id: patientByUserId._id,
          clerkId: patientByUserId.clerkId,
          userId: patientByUserId.userId,
          hasAssignedTherapist: !!patientByUserId.medicalInfo?.assignedTherapist,
          assignedTherapist: patientByUserId.medicalInfo?.assignedTherapist
        } : null
      }
    });

  } catch (error) {
    console.error('Debug patient error:', error);
    return NextResponse.json({ error: 'Internal Server Error', details: error.message }, { status: 500 });
  }
}