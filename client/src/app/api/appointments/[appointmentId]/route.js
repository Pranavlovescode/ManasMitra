import { NextResponse } from 'next/server';
import { getAuth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import Appointment from '@/models/Appointment';

// PATCH - Update appointment (accept/decline/reschedule)
export async function PATCH(request, { params }) {
  try {
    const { userId } = getAuth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { appointmentId } = await params;
    const body = await request.json();
    const { action, scheduledAt, duration, therapistNotes, cancellationReason } = body;

    await connectDB();

    const appointment = await Appointment.findById(appointmentId);
    if (!appointment) {
      return NextResponse.json({ error: 'Appointment not found' }, { status: 404 });
    }

    // Verify user has permission to modify
    if (appointment.therapistClerkId !== userId && appointment.patientClerkId !== userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
    }

    const isTherapist = appointment.therapistClerkId === userId;

    switch (action) {
      case 'accept':
        if (!isTherapist) {
          return NextResponse.json({ error: 'Only therapist can accept' }, { status: 403 });
        }
        appointment.status = 'accepted';
        appointment.modifiedBy = 'therapist';
        appointment.modifiedAt = new Date();
        break;

      case 'decline':
        if (!isTherapist) {
          return NextResponse.json({ error: 'Only therapist can decline' }, { status: 403 });
        }
        appointment.status = 'declined';
        appointment.modifiedBy = 'therapist';
        appointment.modifiedAt = new Date();
        if (therapistNotes) appointment.therapistNotes = therapistNotes;
        break;

      case 'reschedule':
        if (!scheduledAt) {
          return NextResponse.json({ error: 'New scheduled time required' }, { status: 400 });
        }
        if (!appointment.originalScheduledAt) {
          appointment.originalScheduledAt = appointment.scheduledAt;
        }
        appointment.scheduledAt = new Date(scheduledAt);
        appointment.status = 'rescheduled';
        appointment.modifiedBy = isTherapist ? 'therapist' : 'patient';
        appointment.modifiedAt = new Date();
        break;

      case 'update-duration':
        if (!isTherapist) {
          return NextResponse.json({ error: 'Only therapist can update duration' }, { status: 403 });
        }
        if (!duration || duration < 15 || duration > 240) {
          return NextResponse.json({ error: 'Invalid duration (15-240 minutes)' }, { status: 400 });
        }
        appointment.duration = duration;
        appointment.modifiedBy = 'therapist';
        appointment.modifiedAt = new Date();
        break;

      case 'cancel':
        appointment.status = 'cancelled';
        appointment.cancelledBy = isTherapist ? 'therapist' : 'patient';
        appointment.cancelledAt = new Date();
        if (cancellationReason) appointment.cancellationReason = cancellationReason;
        break;

      case 'complete':
        if (!isTherapist) {
          return NextResponse.json({ error: 'Only therapist can mark complete' }, { status: 403 });
        }
        appointment.status = 'completed';
        if (therapistNotes) appointment.therapistNotes = therapistNotes;
        break;

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }

    await appointment.save();

    // Populate before returning
    const populatedAppointment = await Appointment.findById(appointment._id)
      .populate({
        path: 'therapistId',
        populate: { path: 'userId', select: 'firstName lastName email' }
      })
      .populate({
        path: 'patientId',
        populate: { path: 'userId', select: 'firstName lastName email' }
      });

    return NextResponse.json(populatedAppointment);
  } catch (error) {
    console.error('Error updating appointment:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// DELETE - Delete appointment
export async function DELETE(request, { params }) {
  try {
    const { userId } = getAuth(request);
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { appointmentId } = await params;

    await connectDB();

    const appointment = await Appointment.findById(appointmentId);
    if (!appointment) {
      return NextResponse.json({ error: 'Appointment not found' }, { status: 404 });
    }

    // Only allow deletion if user is patient or therapist
    if (appointment.therapistClerkId !== userId && appointment.patientClerkId !== userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 403 });
    }

    await Appointment.findByIdAndDelete(appointmentId);

    return NextResponse.json({ message: 'Appointment deleted successfully' });
  } catch (error) {
    console.error('Error deleting appointment:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
