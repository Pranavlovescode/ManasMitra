import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import Appointment from '@/models/Appointment';
import Patient from '@/models/Patient';
import Therapist from '@/models/Therapist';
import User from '@/models/User';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// GET - Fetch appointments for patient or therapist
export async function GET(request) {
  try {
    const { userId } = await auth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const role = searchParams.get('role'); // 'patient' or 'therapist'

    await connectDB();

    let appointments;
    if (role === 'patient') {
      appointments = await Appointment.find({ patientClerkId: userId })
        .populate({
          path: 'therapistId',
          populate: { path: 'userId', select: 'firstName lastName email' }
        })
        .sort({ scheduledAt: -1 });
    } else if (role === 'therapist') {
      appointments = await Appointment.find({ therapistClerkId: userId })
        .populate({
          path: 'patientId',
          populate: { path: 'userId', select: 'firstName lastName email' }
        })
        .sort({ scheduledAt: -1 });
    } else {
      return NextResponse.json({ error: 'Role parameter required' }, { status: 400 });
    }

    return NextResponse.json(appointments);
  } catch (error) {
    console.error('Error fetching appointments:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// POST - Create new appointment
export async function POST(request) {
  try {
    const { userId } = await auth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { therapistId, scheduledAt, type, duration, patientNotes, meetingType } = body;

    if (!therapistId || !scheduledAt) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    await connectDB();

    // Get or create patient by clerkId
    let patient = await Patient.findOne({ clerkId: userId });
    
    if (!patient) {
      console.log('📝 Patient profile not found, creating one...');
      
      // Get user info from User model
      const user = await User.findOne({ clerkId: userId });
      if (!user) {
        return NextResponse.json({ error: 'User not found' }, { status: 404 });
      }

      // Create patient profile
      patient = new Patient({
        userId: user._id,
        clerkId: userId,
        status: {
          profileComplete: false,
          isActive: true,
        }
      });
      await patient.save();
      console.log('✅ Patient profile created successfully');
    }

    // Resolve therapist by provided identifier.
    // The client may send either the Therapist._id or the User._id of the therapist (from assigned therapist endpoint).
    let therapist = await Therapist.findById(therapistId);
    if (!therapist) {
      // Fallback: try to match by therapist's userId
      therapist = await Therapist.findOne({ userId: therapistId });
    }
    if (!therapist) {
      console.error('THERAPIST NOT FOUND for id:', therapistId);
      return NextResponse.json({ error: 'Therapist not found' }, { status: 404 });
    }

    // Create appointment
    const appointment = new Appointment({
      patientId: patient._id,
      patientClerkId: userId,
      therapistId: therapistId,
      therapistClerkId: therapist.clerkId,
      scheduledAt: new Date(scheduledAt),
      duration: duration || 60,
      type: type || 'individual',
      status: 'pending',
      patientNotes,
      meetingType: meetingType || 'virtual',
    });

    await appointment.save();

    // Populate before returning
    const populatedAppointment = await Appointment.findById(appointment._id)
      .populate({
        path: 'therapistId',
        populate: { path: 'userId', select: 'firstName lastName email' }
      });

    return NextResponse.json(populatedAppointment, { status: 201 });
  } catch (error) {
    console.error('Error creating appointment:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
