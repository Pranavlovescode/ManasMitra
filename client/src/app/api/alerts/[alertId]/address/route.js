import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Alert from '@/models/Alert';

// PATCH /api/alerts/[alertId]/address
export async function PATCH(request, context) {
  try {
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    await connectDB();

    // Get therapist user
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) {
      return NextResponse.json({ error: 'Forbidden: only therapists can address alerts' }, { status: 403 });
    }

    // Get alert ID from params
    const params = await context.params;
    const { alertId } = params;

    // Get notes from request body
    const body = await request.json();
    const { notes } = body;

    // Find and update alert
    const alert = await Alert.findById(alertId);
    if (!alert) {
      return NextResponse.json({ error: 'Alert not found' }, { status: 404 });
    }

    // Verify therapist owns this alert
    if (alert.therapistId.toString() !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden: alert not assigned to therapist' }, { status: 403 });
    }

    // Update alert
    alert.addressed = true;
    alert.addressedAt = new Date();
    alert.addressedBy = therapist._id;
    if (notes) {
      alert.notes = notes;
    }
    await alert.save();

    return NextResponse.json({ 
      success: true, 
      alert: alert.toObject() 
    });

  } catch (error) {
    console.error('PATCH /api/alerts/[alertId]/address error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
