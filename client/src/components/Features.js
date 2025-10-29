import { Heart, Shield, Users, Star, Award, Clock } from 'lucide-react';

export default function Features() {
  const features = [
    {
      icon: Heart,
      title: 'Personalized Care',
      description: 'Tailored mental health tools designed for your unique journey',
      color: 'text-red-500',
      bgColor: 'bg-red-50',
    },
    {
      icon: Shield,
      title: 'HIPAA Compliant',
      description: 'Bank-level security ensuring your privacy and confidentiality',
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
    },
    {
      icon: Users,
      title: 'Professional Network',
      description: 'Connect with licensed therapists and mental health professionals',
      color: 'text-green-500',
      bgColor: 'bg-green-50',
    },
    {
      icon: Star,
      title: 'Evidence-Based',
      description: 'Tools and techniques backed by clinical research and proven methods',
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-50',
    },
    {
      icon: Award,
      title: 'Quality Assured',
      description: 'Continuously improved based on user feedback and clinical outcomes',
      color: 'text-purple-500',
      bgColor: 'bg-purple-50',
    },
    {
      icon: Clock,
      title: '24/7 Available',
      description: 'Access your mental health tools and resources anytime, anywhere',
      color: 'text-indigo-500',
      bgColor: 'bg-indigo-50',
    },
  ];

  return (
    <section className="py-16 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Why Choose ManasMitra?
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Experience comprehensive mental health support with cutting-edge technology 
            and compassionate care.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="group p-6 rounded-2xl border border-gray-100 hover:border-gray-200 hover:shadow-lg transition-all duration-300"
            >
              <div className={`w-12 h-12 ${feature.bgColor} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-200`}>
                <feature.icon className={`w-6 h-6 ${feature.color}`} />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}