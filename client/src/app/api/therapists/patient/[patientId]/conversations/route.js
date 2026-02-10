import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import Conversation from '@/models/Conversation';

// GET /api/therapists/patient/[patientId]/conversations
export async function GET(request, context) {
  try {
    const { userId } = await auth();
    
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    // In Next.js 15+, params might need to be awaited
    const params = await context.params;
    const { patientId } = params;

    // Get therapist user
    const therapistUser = await User.findOne({ clerkId: userId });
    if (!therapistUser || therapistUser.role !== 'therapist') {
      return NextResponse.json({ error: 'Forbidden:  only therapists can access this endpoint' }, { status: 403 });
    }

    // Get patient and verify assignment
    const patient = await Patient.findById(patientId);
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    // Verify therapist is assigned to this patient
    const assignedTherapistId = patient.medicalInfo?.assignedTherapist || patient.status?.assignedTherapist;
    if (assignedTherapistId && assignedTherapistId.toString() !== therapistUser._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    // Fetch conversations by patient's Clerk ID
    const conversations = await Conversation.find({ userId: patient.clerkId })
      .sort({ createdAt: -1 })
      .limit(100)
      .lean();

    return NextResponse.json(conversations);
  } catch (error) {
    console.error('Therapist conversations GET error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
