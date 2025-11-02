import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Patient from '@/models/Patient';
import mongoose from 'mongoose';

// Define Mood model locally (same as in /api/mood)
const MoodSchema = new mongoose.Schema({
  userId: { type: String, required: true, index: true },
  mood: { type: String, required: true, enum: ['sad', 'neutral', 'happy', 'excited', 'loved'] },
  intensity: { type: Number, required: true, min: 1, max: 10 },
  notes: { type: String, default: '' },
  date: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
}, { timestamps: true });

MoodSchema.index({ userId: 1, date: 1 });
const Mood = mongoose.models.Mood || mongoose.model('Mood', MoodSchema);

// GET /api/therapists/patient/[patientId]/moods
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

    const moods = await Mood.find({ userId: patient.clerkId }).sort({ timestamp: -1 }).limit(100).lean();
    return NextResponse.json(moods);
  } catch (error) {
    console.error('Therapist moods GET error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
