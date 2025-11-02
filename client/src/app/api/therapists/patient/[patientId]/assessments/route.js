import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import Assessment from '@/models/Assessment';

// GET /api/therapists/patient/[patientId]/assessments
export async function GET(request, { params }) {
  try {
    const { userId } = await auth(request);
    if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    await connectDB();

    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });

    const { patientId } = await params;
    const patient = await Patient.findById(patientId).select('clerkId medicalInfo.assignedTherapist').populate('userId', 'firstName lastName');
    if (!patient) return NextResponse.json({ error: 'Patient not found' }, { status: 404 });

    if (patient.medicalInfo?.assignedTherapist?.toString() !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    const assessments = await Assessment.find({ userId: patient.clerkId }).sort({ date: -1 }).limit(100).lean();
    return NextResponse.json(assessments);
  } catch (error) {
    console.error('Therapist assessments GET error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
