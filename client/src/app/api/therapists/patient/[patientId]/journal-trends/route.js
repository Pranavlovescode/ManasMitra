import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import Journal from '@/models/Journal';
import { computeJournalTrends } from '@/lib/analysis';

// GET /api/therapists/patient/[patientId]/journal-trends
// Returns aggregated trends for journals if therapist is assigned
export async function GET(request, { params }) {
  try {
    const { userId } = await auth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { patientId } = await params;

    // Verify therapist
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    // Verify patient assignment
    const patient = await Patient.findById(patientId).select('clerkId medicalInfo.assignedTherapist');
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const assigned = patient.medicalInfo?.assignedTherapist?.toString();
    if (!assigned || assigned !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    // Optional window filtering
    const { searchParams } = new URL(request.url);
    const since = searchParams.get('since'); // ISO date
    const until = searchParams.get('until'); // ISO date

    const query = { userId: patient.clerkId };
    if (since || until) {
      query.createdAt = {};
      if (since) query.createdAt.$gte = new Date(since);
      if (until) query.createdAt.$lte = new Date(until);
    }

    const journals = await Journal.find(query).sort({ createdAt: 1 }).lean();
    const trends = computeJournalTrends(journals);

    return NextResponse.json({
      patientId,
      ...trends,
    });
  } catch (error) {
    console.error('Therapist journal trends error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
