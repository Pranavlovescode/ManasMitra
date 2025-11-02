import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import Journal from '@/models/Journal';

// GET /api/therapists/patient/[patientId]/journals
// Returns journal entries for a patient if the requesting user is the assigned therapist
export async function GET(request, { params }) {
  try {
    const { userId } = await auth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    const { patientId } = await params;

    // Verify therapist identity and role
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) {
      return NextResponse.json({ error: 'Access denied. Therapist account required.' }, { status: 403 });
    }

    // Load patient and verify assignment to therapist
    const patient = await Patient.findById(patientId).select('clerkId medicalInfo.assignedTherapist');
    if (!patient) {
      return NextResponse.json({ error: 'Patient not found' }, { status: 404 });
    }

    const assigned = patient.medicalInfo?.assignedTherapist?.toString();
    if (!assigned || assigned !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: patient not assigned to therapist' }, { status: 403 });
    }

    // Pagination params
    const { searchParams } = new URL(request.url);
    const page = Math.max(parseInt(searchParams.get('page') || '1', 10), 1);
    const limit = Math.min(Math.max(parseInt(searchParams.get('limit') || '20', 10), 1), 100);
    const skip = (page - 1) * limit;

    // Fetch journals by patient's Clerk ID
    const query = { userId: patient.clerkId };
    const [items, total] = await Promise.all([
      Journal.find(query).sort({ createdAt: -1 }).skip(skip).limit(limit).lean(),
      Journal.countDocuments(query),
    ]);

    return NextResponse.json({
      items,
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit) || 1,
    });
  } catch (error) {
    console.error('Therapist journals GET error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
