import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import connectDB from '@/lib/mongodb';
import User from '@/models/User';
import Alert from '@/models/Alert';

// GET /api/alerts?therapistId={clerkId}
export async function GET(request) {
  try {
    const { userId } = await auth(request);
    if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    await connectDB();

    // Verify therapist identity
    const therapist = await User.findOne({ clerkId: userId, role: 'therapist' });
    if (!therapist) return NextResponse.json({ error: 'Access denied' }, { status: 403 });

    const { searchParams } = new URL(request.url);
    const paramTherapistId = searchParams.get('therapistId'); // Clerk ID (from UI)
    if (paramTherapistId && paramTherapistId !== userId) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const alerts = await Alert.find({ therapistId: therapist._id, dismissed: false })
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();

    return NextResponse.json(alerts);
  } catch (error) {
    console.error('GET /api/alerts error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
