import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Alert from '@/models/Alert';

// DELETE /api/alerts/[alertId]
export async function DELETE(request, { params }) {
  try {
    const { userId } = await auth(request);
    if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    await connectDB();
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) return NextResponse.json({ error: 'Access denied' }, { status: 403 });

    const { alertId } = await params;
    const alert = await Alert.findById(alertId);
    if (!alert) return NextResponse.json({ error: 'Not found' }, { status: 404 });

    if (alert.therapistId.toString() !== therapist._id.toString()) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    alert.dismissed = true;
    await alert.save();

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('DELETE /api/alerts/[alertId] error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
